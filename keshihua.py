import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw

df = pd.read_csv('./traj.csv')

# 将coordinates列转换为经度和纬度两列
df[['longitude', 'latitude']] = pd.DataFrame(df['coordinates'].apply(lambda x: eval(x)).tolist(), index=df.index)
# 根据traj_id分组，提取出不同的轨迹
traj_list = []
for traj_id, group in df.groupby('traj_id'):
    traj_coords = [(row['longitude'], row['latitude']) for _, row in group.iterrows()]
    traj_list.append(traj_coords)

from sklearn.cluster import DBSCAN
from fastdtw import fastdtw
import numpy as np
from tqdm import tqdm


def directed_dtw(traj1, traj2):
    n = len(traj1)
    m = len(traj2)

    # 初始化距离矩阵
    D = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        D[i, 0] = np.inf
    for j in range(1, m + 1):
        D[0, j] = np.inf

    # 计算距离矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt((traj1[i - 1][0] - traj2[j - 1][0]) ** 2 + (traj1[i - 1][1] - traj2[j - 1][1]) ** 2)
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return D[n, m]


# 自定义距离计算函数
def calculate_dis(traj1, traj2):
    traj1_array = np.array(traj1)
    traj2_array = np.array(traj2)
    # distance = directed_dtw(traj1_array, traj2_array)
    distance, path = fastdtw(traj1_array, traj2_array)
    return distance


# 轨迹数据
trajs = traj_list

# 将轨迹数据转换为距离矩阵
dist_matrix = np.zeros((len(trajs), len(trajs)))
for i in tqdm(range(len(trajs))):
    for j in range(i + 1, len(trajs)):
        dist_matrix[i][j] = calculate_dis(trajs[i], trajs[j])
        dist_matrix[j][i] = dist_matrix[i][j]

# 使用DBSCAN进行聚类
dbscan = DBSCAN(metric='precomputed', eps=16, min_samples=1)
dbscan.fit(dist_matrix)

# 输出聚类结果
labels = dbscan.labels_
print(labels)