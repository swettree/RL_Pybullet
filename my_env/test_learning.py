import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 改进的生成随机点的函数
def generate_random_point(num_points):
    # 生成均匀分布的极角 θ 在 [0, π] 范围内
    theta = torch.acos(2 * torch.rand(num_points) - 1)
    # 生成均匀分布的方位角 φ 在 [0, 2π] 范围内
    phi = torch.rand(num_points) * 2 * torch.pi

    # 将球坐标转换为笛卡尔坐标
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    # 生成的方向向量
    point_hat = torch.stack([x, y, z], dim=1)

    return point_hat

# 生成1000个点
num_points = 1000
points = generate_random_point(num_points)

# 将点转换为numpy数组以便于绘图
points_np = points.numpy()

# 绘制3D图
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Randomly Generated Points on Unit Sphere')
plt.show()