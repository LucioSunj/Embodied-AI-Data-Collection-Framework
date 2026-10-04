import os
import glob
import subprocess
import pandas as pd

def process_parquet_to_10hz(data_dir):
    """处理 Parquet 轨迹文件：降采样至 10Hz，并将 action 替换为下一帧的 state"""
    search_path = os.path.join(data_dir, "**", "*.parquet")
    file_list = glob.glob(search_path, recursive=True)
    
    if not file_list:
        print("未在数据目录找到任何 .parquet 文件")
        return

    print(f"找到 {len(file_list)} 个 Parquet 文件，开始处理...")

    for file_path in file_list:
        try:
            df = pd.read_parquet(file_path)
            
            # 降采样 (30Hz -> 10Hz)
            df = df.iloc[::3].copy().reset_index(drop=True)
            
            
            # 2. 修正 frame_index 显式列
            # 将外部连续的隐式行索引直接赋给 frame_index 列
            df['frame_index'] = df.index
            # 将 action 替换为下一帧的 qpos (state)
            #new_actions = df['qpos'].shift(-1)
            
            # 最后一帧处理
            #new_actions.iloc[-1] = df['qpos'].iloc[-1]
            
            # 更新并写回
            #df['action'] = new_actions
            df.to_parquet(file_path, index=False, engine='pyarrow')
            
        except Exception as e:
            print(f"处理 Parquet 文件 {file_path} 时出错: {e}")
    print("Parquet 文件处理完成。")

def process_images_to_10hz(images_dir):
    """处理图像序列：保留 0, 3, 6... 帧，并重命名为连续序号 0, 1, 2..."""
    camera_dirs = glob.glob(os.path.join(images_dir, "*", "*"))
    
    if not camera_dirs:
        print("未找到图像目录，请检查路径。")
        return

    for cam_dir in camera_dirs:
        if not os.path.isdir(cam_dir):
            continue
            
        img_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
        if not img_files:
            continue
            
        print(f"正在处理图像目录: {cam_dir} (共 {len(img_files)} 张)")
        
        for original_idx, img_path in enumerate(img_files):
            if original_idx % 3 == 0:
                new_idx = original_idx // 3
                new_name = f"{new_idx:06d}.jpg"
                new_path = os.path.join(cam_dir, new_name)
                
                if img_path != new_path:
                    os.rename(img_path, new_path)
            else:
                os.remove(img_path)
    print("图像序列处理完成。")

def process_videos_to_10hz(videos_dir):
    """处理视频：使用 ffmpeg 将 30fps 的视频转为 10fps"""
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    
    if not video_files:
        print("未找到视频文件，请检查路径。")
        return

    for vid_path in video_files:
        print(f"正在转换视频帧率: {os.path.basename(vid_path)}")
        temp_out = vid_path.replace(".mp4", "_10hz_temp.mp4")
        
        cmd = [
            "ffmpeg", "-y", "-i", vid_path, 
            "-filter:v", "fps=10", 
            "-loglevel", "error", 
            temp_out
        ]
        
        try:
            subprocess.run(cmd, check=True)
            os.replace(temp_out, vid_path)
        except subprocess.CalledProcessError as e:
            print(f"视频 {vid_path} 转换失败: {e}")
            if os.path.exists(temp_out):
                os.remove(temp_out)
    print("视频文件处理完成。")

if __name__ == "__main__":
    # 基础路径配置
    base_dirs = ["/home/lsk/openpi/Experiments/fr3_gello_lerobot_pickall/",
                #"/home/lsk/openpi/Experiments/carrot_pepper_corn_all/lerobot_datasets_carrot/",
                #"/home/lsk/openpi/Experiments/carrot_pepper_corn_all/lerobot_datasets_corn/",
                #"/home/lsk/openpi/Experiments/carrot_pepper_corn_all/lerobot_datasets_green_pepper/"
                ]
    for base_dir in base_dirs:
        data_dir = os.path.join(base_dir, "data")
        images_dir = os.path.join(base_dir, "images")
        videos_dir = os.path.join(base_dir, "videos")
        
        print("========== 开始多模态数据集对齐任务 ==========\n")
        
        print("--- 阶段 1: 处理数值轨迹 (Parquet) ---")
        process_parquet_to_10hz(data_dir)
        print("\n")
        
        print("--- 阶段 2: 处理图像序列 (Images) ---")
        process_images_to_10hz(images_dir)
        print("\n")
        
        print("--- 阶段 3: 处理视频录像 (Videos) ---")
        print("提示：此过程依赖 ffmpeg，且可能需要较长时间。")
        process_videos_to_10hz(videos_dir)
        print("\n")
        
    print("========== 所有任务处理完毕，数据集已对齐至 10Hz ==========")