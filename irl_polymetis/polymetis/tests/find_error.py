import pandas as pd
import glob

files = glob.glob("/home/lsk/irl_polymetis/lerobot_datasets_green_pepper/data/*.parquet")
for f in files:
    df = pd.read_parquet(f)
    # 打印每一份文件的第四位动作的最大值
    print(f"文件: {f}, Action[3] 最大值: {df['action'].str[3].max()}")