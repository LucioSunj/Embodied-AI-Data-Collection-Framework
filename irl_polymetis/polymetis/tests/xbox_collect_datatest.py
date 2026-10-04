import time
import threading
import numpy as np
import h5py
import pyrealsense2 as rs
import os
import xbox_fr3_data
import cv2

class MulticamDataCollector:
    def __init__(self, camera_serials, save_dir="vla_datasets", chunk_size=50, chunk_stride=10):
        """
        :param chunk_size: (H) 每个图像对应的未来动作步数（例如 50 步）
        :param chunk_stride: (Delta) 在 1000Hz 缓存中提取动作的步长 (10 代表降采样到 100Hz)
        """
        self.camera_serials = camera_serials
        self.save_dir = save_dir
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.episode_count = 0
        self.is_recording = False
        self.is_running = True

        
        # 数据缓存区 (1000Hz 收集会导致 robot_buffer 极大，使用 list 配合 append 依然高效)
        self.image_buffer = []
        self.robot_buffer = []
        
        self.pipelines = {}
        self.aligns = {}
        self._init_cameras()
        self.cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.cam_thread.start()
        

    def _init_cameras(self):
        """初始化 RealSense 阵列，并包含底层硬件强制复位"""
        ctx = rs.context()
        
        # 1. 强制复位所有目标硬件 (极其重要)
        print("🔄 正在向 RealSense 发送底层硬件复位指令 (需等待约3秒)...")
        for dev in ctx.query_devices():
            if dev.get_info(rs.camera_info.serial_number) in self.camera_serials:
                dev.hardware_reset()
        
        # 给硬件足够的时间重新枚举 USB 设备
        time.sleep(3.5) 
        
        # 2. 重新启动管线
        for serial in self.camera_serials:
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                #pipeline.start(config)
                pipeline_profile = pipeline.start(config)
                device = pipeline_profile.get_device()
                for sensor in device.query_sensors():
                    if sensor.is_depth_sensor():
                        try:
                            if sensor.supports(rs.option.emitter_enabled):
                                sensor.set_option(rs.option.emitter_enabled, 0)
                                print(f"相机[{serial}]已关闭红外激光")
                        except Exception as e:
                            pass
                self.pipelines[serial] = pipeline
                self.aligns[serial] = rs.align(rs.stream.color)
                
                # 预热几帧，让相机的自动曝光(AE)稳定下来
                for _ in range(10):
                    pipeline.wait_for_frames(timeout_ms=2000)
                    
                print(f"✅ 相机 [{serial}] 硬件复位并启动就绪 (30Hz)")
            except Exception as e:
                print(f"❌ 相机 [{serial}] 启动失败，请检查连线！报错: {e}")

    def start_episode(self):
        if self.is_recording:
            return
            
        self.image_buffer.clear()
        self.robot_buffer.clear()
        self.is_recording = True
        print(f"\n🔴 [正在录制] Episode_{self.episode_count:04d} (1000Hz 动作采集中...)")
        
        #self.cam_thread = threading.Thread(target=self._camera_worker)
        #self.cam_thread.start()

    def stop_and_save_episode(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        #self.cam_thread.join()
        #cv2.destroyAllWindows()
        print(f"\n⏹️ [录制结束] 1000Hz 缓存容量: {len(self.robot_buffer)}，执行 Chunking 对齐打包...")
        self._save_to_hdf5()
        self.episode_count += 1

    def record_robot_step(self, qpos, action):
        """1000Hz 极速写入探针，绝对不能有任何耗时操作"""
        if self.is_recording:
            self.robot_buffer.append({
                'timestamp': time.time(),
                'qpos': np.array(qpos, dtype=np.float32),
                'action': np.array(action, dtype=np.float32)
            })

    
    def _camera_worker(self):
        
        frames_processed = 0 
        last_fps_time = time.time()
        
        # 新增：为每个序列号创建一个独立的帧数计数器
        cam_frame_counters = {serial: 0 for serial in self.camera_serials}
        
        while self.is_running:
            frame_data = {'timestamp': time.time()}
            current_time = time.time()
            
            # 1. 尝试从所有相机拉取画面
            for serial, pipeline in self.pipelines.items():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        # 加上 .copy()，强制深拷贝！彻底解放底层缓存池！
                        frame_data[f'cam_{serial}'] = np.asanyarray(color_frame.get_data()).copy()
                        # 当前相机成功吐出一帧，独立计数器 +1
                        cam_frame_counters[serial] += 1
                except Exception:
                    pass # 这里的异常不抛出，交给后面的逻辑统一处刑
            
            # 2. 严格判定这一帧是否完整集齐了所有相机
            # 必须等于 序列号总数 + 1（那个1是时间戳）
            is_frame_valid = (len(frame_data) == len(self.camera_serials) + 1)
            
            # 3. 核心分发逻辑
            if is_frame_valid:
                frames_processed += 1 # 只要硬件吐出了有效帧，整体同步FPS就+1
                if self.is_recording:
                    self.image_buffer.append(frame_data) # 只有录制期间才存入内存
            else:
                # 如果没集齐，且当前正在录制，我们要打印到底缺了谁的数据
                if self.is_recording:
                    expected_cams = set([f'cam_{s}' for s in self.camera_serials])
                    actual_cams = set(frame_data.keys()) - {'timestamp'}
                    missing = expected_cams - actual_cams
                    print(f"⚠️ [录制中] 丢弃残缺帧：未收到相机 {missing} 的数据")
            
            # 4. 每秒结算一次 FPS (独立 FPS 与 整体同步 FPS)
            if current_time - last_fps_time >= 1.0:
                elapsed = current_time - last_fps_time
                
                # 计算并拼接每个相机的独立帧率
                individual_fps_list = []
                for serial in self.camera_serials:
                    cam_fps = cam_frame_counters[serial] / elapsed
                    individual_fps_list.append(f"[{serial}]: {cam_fps:.1f}帧")
                    cam_frame_counters[serial] = 0 # 算完清零
                
                cam_fps_str = " | ".join(individual_fps_list)
                
                # 计算整体完美同步的帧率（决定了最终能存下多少数据）
                actual_sync_fps = frames_processed / elapsed
                
                # 状态栏动态显示
                status_text = "🔴 录制中" if self.is_recording else "⚪ 待机中"
                print(f"[{status_text}] {cam_fps_str} || 整体同步: {actual_sync_fps:.1f} FPS | 缓存: {len(self.image_buffer)}")
                
                # 重置计数器
                frames_processed = 0
                last_fps_time = current_time
    def _save_to_hdf5(self):
        file_name = os.path.join(self.save_dir, f"episode_{self.episode_count:04d}.h5")
        
        # 预先提取 1000Hz 的时间戳数组，利用 numpy 的向量化搜索大幅提升性能
        robot_timestamps = np.array([step['timestamp'] for step in self.robot_buffer])
        
        with h5py.File(file_name, 'w') as root:
            obs_grp = root.create_group('observations')
            images_grp = obs_grp.create_group('images')
            
            aligned_action_chunks = []
            aligned_joints = []
            valid_image_indices = []
            
            # 遍历 30Hz 的图像帧
            for img_idx, img_frame in enumerate(self.image_buffer):
                cam_ts = img_frame['timestamp']
                
                # 向量化寻找最近邻索引 (i*)
                closest_idx = np.argmin(np.abs(robot_timestamps - cam_ts))
                
                # Action Chunking 边界检查：确保未来还有足够的动作序列可以提取
                required_future_steps = self.chunk_size * self.chunk_stride
                if closest_idx + required_future_steps > len(self.robot_buffer):
                    # 如果这帧图像对应的未来动作不够了（发生在录制末尾），则舍弃该图像帧
                    continue
                
                valid_image_indices.append(img_idx)
                
                # 提取当前状态
                aligned_joints.append(self.robot_buffer[closest_idx]['qpos'])
                
                # 提取未来 H 步动作轨迹作为一个 Chunk
                chunk = []
                for step in range(self.chunk_size):
                    future_idx = closest_idx + step * self.chunk_stride
                    chunk.append(self.robot_buffer[future_idx]['action'])
                
                aligned_action_chunks.append(chunk)
            
            # 写入图像 (只存有效且未来动作完整的帧)
            for serial in self.camera_serials:
                img_array = np.array([self.image_buffer[idx][f'cam_{serial}'] for idx in valid_image_indices])
                images_grp.create_dataset(f'cam_{serial}', data=img_array, compression="gzip")
                
            obs_grp.create_dataset('qpos', data=np.array(aligned_joints))
            
            # 最终的 action 形状将是 [N_valid_frames, chunk_size, action_dim]
            root.create_dataset('action', data=np.array(aligned_action_chunks))
            
            print(f"💾 Chunking 打包成功！文件: {file_name}")
            print(f"📊 数据维度: 图像={len(valid_image_indices)}帧, 动作矩阵={np.array(aligned_action_chunks).shape}")


if __name__ == '__main__':
    TARGET_CAMERAS = ['352122270841', '348122070707'] 
    #TARGET_CAMERAS = ['352122270841']
    # chunk_size=50, chunk_stride=10 意味着：
    # 从 1000Hz 数据中，每隔 10ms 抽一帧，总共抽 50 帧未来动作。
    # 相当于模型学习的是一段 0.5 秒钟的平滑降采样轨迹。
    collector = MulticamDataCollector(camera_serials=TARGET_CAMERAS, chunk_size=50, chunk_stride=10)
    
    xbox_fr3_data.DATA_COLLECTOR = collector
    
    print("\n🚀 1000Hz 异步采集后端初始化完毕！")
    xbox_fr3_data.run_xbox_teleop()
