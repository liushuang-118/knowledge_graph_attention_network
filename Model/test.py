import pandas as pd
from collections import defaultdict
from datetime import datetime

# ----------------------------
# 配置
# ----------------------------
BASE_DIR = r"D:\Thesis_Project\Models\KGAT\Data\amazon-beauty"
PATH_FILE = f"{BASE_DIR}/kgat_topk_paths.txt"
TIME_TRAIN_FILE = f"{BASE_DIR}/time_train.csv"
TIME_TEST_FILE = f"{BASE_DIR}/time_test.csv"
BETA_LIR = 0.5  # temporal decay factor

# ----------------------------
# 加载时间数据
# ----------------------------
train_df = pd.read_csv(TIME_TRAIN_FILE)
test_df = pd.read_csv(TIME_TEST_FILE)
all_time_df = pd.concat([train_df, test_df], ignore_index=True)

# 转换时间为 datetime
all_time_df['PURCHASE_Time'] = pd.to_datetime(all_time_df['PURCHASE_Time'], errors='coerce')
all_time_df = all_time_df.dropna(subset=['PURCHASE_Time'])
min_date = all_time_df['PURCHASE_Time'].min()
all_time_df['PURCHASE_Time_days'] = (all_time_df['PURCHASE_Time'] - min_date).dt.days

# 构建用户-商品时间字典
user_item_time = defaultdict(dict)
for row in all_time_df.itertuples(index=False):
    user_item_time[row.UID][row.PID] = row.PURCHASE_Time_days

# 记录每个用户的最近购买时间（用于填充没有交互的产品）
user_max_time = {uid: max(times.values()) for uid, times in user_item_time.items()}

# ----------------------------
# 解析路径文件
# ----------------------------
user_paths = defaultdict(list)
with open(PATH_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

current_user = None
current_path = []

for line in lines:
    line = line.strip()
    if line.startswith("User"):
        current_user = int(line.split()[1])
        current_path = []
    elif line.startswith("Path") or line.startswith("No path found"):
        if current_path:
            user_paths[current_user].append(current_path)
        current_path = []
    elif line.startswith("("):
        # 解析 (head, rel, tail)
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                h = parts[0].strip()
                t = parts[2].strip().split(')')[0]
                # 把 ID 转成整数
                h_id = int(h.split()[-1]) if 'User' in h else int(h)
                t_id = int(t)
                current_path.append((h_id, t_id))
            except:
                continue

# ----------------------------
# 计算 LIR
# ----------------------------
def compute_lir(uid, path):
    """
    按照 EWMA 计算路径的 LIR
    如果用户没有交互某个商品，用最近交互时间填充
    """
    times = []
    max_time = user_max_time.get(uid, 0)  # 默认填充最近时间
    for h_id, t_id in path:
        # 取用户交互时间，如果没有则用 max_time
        t = user_item_time.get(uid, {}).get(t_id, max_time)
        times.append(t)
    
    if not times:
        return 0.0
    
    # 排序并按 EWMA 递归计算
    times.sort()
    lir = times[0]
    for x in times[1:]:
        lir = (1 - BETA_LIR) * lir + BETA_LIR * x

    # 归一化到 0-1
    lir_norm = lir / max_time if max_time > 0 else 0.0
    return lir_norm

# ----------------------------
# 计算所有用户平均 LIR
# ----------------------------
user_lir = {}
for uid, paths in user_paths.items():
    lir_scores = [compute_lir(uid, path) for path in paths]
    if lir_scores:
        user_lir[uid] = sum(lir_scores) / len(lir_scores)

# 输出平均 LIR
if user_lir:
    avg_lir = sum(user_lir.values()) / len(user_lir)
    print(f"[INFO] 平均 LIR (0~1): {avg_lir:.4f}")
else:
    print("[WARN] 没有可计算 LIR 的用户")
