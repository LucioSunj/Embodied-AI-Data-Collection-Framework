import time
import threading
import numpy as np
import pyrealsense2 as rs
import os
import cv2
import pandas as pd  
import xbox_fr3_data

class MulticamLeRobotCollector:
    def __init__(self, camera_serials, task_instruction, save_dir="lerobot_datasets", fps=30):
        self.camera_serials = camera_serials
        self.task_instruction = task_instruction  # 言指令
        self.save_dir = save_dir
        self.fps = fps
        
        # LeRobot 标准目录结构
        self.video_dir = os.path.join(self.save_dir, "videos")
        self.data_dir = os.path.join(self.save_dir, "data")
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
            
        self.episode_count = 38
        self.is_recording = False
        self.is_running = True

        self.image_buffer = []
        self.robot_buffer = []
        
        self.pipelines = {}
        self._init_cameras()
        self.cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.cam_thread.start()

    def _init_cameras(self):
        # (保持原有的硬件复位和管线初始化逻辑，此处略作精简以突出重点)
        ctx = rs.context()
        print("🔄 正在向 RealSense 发送底层硬件复位指令...")
        for dev in ctx.query_devices():
            if dev.get_info(rs.camera_info.serial_number) in self.camera_serials:
                dev.hardware_reset()
        time.sleep(3.5) 
        
        for serial in self.camera_serials:
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)
                
                pipeline_profile = pipeline.start(config)
                device = pipeline_profile.get_device()
                for sensor in device.query_sensors():
                    if sensor.is_depth_sensor() and sensor.supports(rs.option.emitter_enabled):
                        sensor.set_option(rs.option.emitter_enabled, 0)
                
                self.pipelines[serial] = pipeline
                for _ in range(10): pipeline.wait_for_frames(timeout_ms=2000)
                print(f"✅ 相机 [{serial}] 硬件复位并启动就绪 ({self.fps}Hz)")
            except Exception as e:
                print(f"❌ 相机 [{serial}] 启动失败: {e}")

    def start_episode(self):
        if self.is_recording: return
        self.image_buffer.clear()
        self.robot_buffer.clear()
        self.is_recording = True
        print(f"\n🔴 [录制中] 任务: '{self.task_instruction}' | Episode_{self.episode_count:04d}")

    def stop_and_save_episode(self):
        if not self.is_recording: return
        self.is_recording = False
        print(f"\n⏹️ [录制结束] 执行 LeRobot 格式 (MP4 + Parquet) 对齐打包...")
        self._save_to_lerobot_format()
        self.episode_count += 1

    def record_robot_step(self, qpos, action):
        if self.is_recording:
            gripper_width = action[-1]
            #print(gripper_width)
            gripper_binary_state = 1.0 if gripper_width > 0.05 else 0.0
            full_qpos = np.append(qpos, gripper_binary_state)
            self.robot_buffer.append({
                'timestamp': time.time(),
                'qpos': np.array(full_qpos, dtype=np.float32),
                'action': np.array(action, dtype=np.float32)
            })

    def _camera_worker(self):
        # (保持原有的严格帧同步和丢帧检测逻辑，这部分你写得非常好)
        frames_processed = 0 
        last_fps_time = time.time()
        cam_frame_counters = {serial: 0 for serial in self.camera_serials}
        
        while self.is_running:
            frame_data = {'timestamp': time.time()}
            current_time = time.time()
            
            for serial, pipeline in self.pipelines.items():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=100) # 缩短 timeout 防止死锁
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        frame_data[f'cam_{serial}'] = np.asanyarray(color_frame.get_data()).copy()
                        cam_frame_counters[serial] += 1
                except Exception:
                    pass 
            
            is_frame_valid = (len(frame_data) == len(self.camera_serials) + 1)
            
            if is_frame_valid:
                frames_processed += 1 
                if self.is_recording:
                    self.image_buffer.append(frame_data) 
            
            if current_time - last_fps_time >= 1.0:
                elapsed = current_time - last_fps_time
                actual_sync_fps = frames_processed / elapsed
                if self.is_recording:
                    print(f"[录制中] 整体同步: {actual_sync_fps:.1f} FPS | 缓存帧数: {len(self.image_buffer)}", end='\r')
                frames_processed = 0
                last_fps_time = current_time

    def _save_to_lerobot_format(self):
        if len(self.image_buffer) == 0 or len(self.robot_buffer) == 0:
            print("⚠️ 警告：缓存为空，跳过保存。")
            return

        robot_timestamps = np.array([step['timestamp'] for step in self.robot_buffer])
        
        aligned_data = {
            'frame_index': [],
            'timestamp': [],
            'qpos': [],
            'action': [],
            'language_instruction': []
        }
        
        valid_image_indices = []
        
        # 1. 向量化时间戳对齐 (转换为单步对应，抛弃预打包 Chunk)
        for img_idx, img_frame in enumerate(self.image_buffer):
            cam_ts = img_frame['timestamp']
            closest_idx = np.argmin(np.abs(robot_timestamps - cam_ts))
            
            valid_image_indices.append(img_idx)
            
            aligned_data['frame_index'].append(len(valid_image_indices) - 1)
            aligned_data['timestamp'].append(cam_ts)
            # LeRobot 通常需要 list 或一维 numpy 数组存储状态和动作
            aligned_data['qpos'].append(self.robot_buffer[closest_idx]['qpos'].tolist())
            aligned_data['action'].append(self.robot_buffer[closest_idx]['action'].tolist())
            aligned_data['language_instruction'].append(self.task_instruction)

        # 2. 保存视频 (MP4 格式)
        for serial in self.camera_serials:
            vid_filename = f"cam_{serial}_episode_pickup_corn_{self.episode_count:06d}.mp4"
            vid_path = os.path.join(self.video_dir, vid_filename)
            
            # 使用 mp4v 或 avc1 编码
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_path, fourcc, self.fps, (640, 480))
            
            for idx in valid_image_indices:
                
                out.write(self.image_buffer[idx][f'cam_{serial}'])
            
            out.release()
            print(f"🎬 视频已保存: {vid_path}")

        # 3. 保存本体状态与语言指令 (Parquet 格式)
        df = pd.DataFrame(aligned_data)
        parquet_filename = f"episode_pickup_corn_{self.episode_count:06d}.parquet"
        parquet_path = os.path.join(self.data_dir, parquet_filename)
        df.to_parquet(parquet_path, engine='pyarrow')
        
        print(f"📊 状态与指令已保存: {parquet_path}")
        print(f"✅ Episode {self.episode_count} 包含 {len(valid_image_indices)} 帧有效数据。")


if __name__ == '__main__':
    TARGET_CAMERAS = ['352122270841', '348122070707','347622075736'] 
    
    print("========================================")
    print("🤖 具身智能数据采集系统 (LeRobot 格式)")
    print("========================================")
    
    # 交互式输入当前批次的语言指令
    task_text = input("✏️ 请输入当前采集任务的语言指令 (例如 'Pick up the carrot and put it in the box'): \n> ")
    if not task_text.strip():
        task_text = "manipulate objects" # 默认 fallback
        
    collector = MulticamLeRobotCollector(
        camera_serials=TARGET_CAMERAS, 
        task_instruction=task_text,
        save_dir="lerobot_datasets"
    )
    
    xbox_fr3_data.DATA_COLLECTOR = collector
    
    print("\n🚀 后端初始化完毕！按动手柄按键开始遥操作与录制。")
    xbox_fr3_data.run_xbox_teleop()