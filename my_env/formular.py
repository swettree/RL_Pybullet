
import numpy as np
import matplotlib.pyplot as plt
import csv

""" 定义函数 """
# def tanh_decay(d):
#     return 1-np.tanh(10.0 * d / 5)

# def exp_decay(d):
#     return np.exp(-10*d)

# def tanh_nur(d):
#     return np.tanh(d)

# # 生成 d 值
# d_values = np.linspace(0, 1, 400)
# tanh_values = tanh_decay(d_values)
# exp_values = exp_decay(d_values)
# tanh_nur_value = tanh_nur(d_values)
# # 绘制函数图像
# plt.figure(figsize=(10, 6))
# plt.plot(d_values, tanh_values, label='1 - tanh(10.0 * d / 3.0)')
# plt.plot(d_values, exp_values, label='exp(-d / 0.5)')
# plt.plot(d_values, tanh_nur_value, label='np.tanh(d)')
# plt.title('Comparison of 1 - tanh(10.0 * d / 1.0) and exp(-d / 0.5)')
# plt.xlabel('d')
# plt.ylabel('Value')
# plt.grid(True)
# plt.legend()
# plt.show()

"""画轨迹"""
def read_coordinates_from_csv(file_path):
    coordinates = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            coordinates.append([float(value) for value in row])
    
    # 确保坐标数量是3的倍数
    if len(coordinates) % 3 != 0:
        raise ValueError("The number of rows in the CSV file is not a multiple of 3.")
    
    # 将每三行作为X, Y, Z坐标
    x_coords = coordinates[0::3]
    y_coords = coordinates[1::3]
    z_coords = coordinates[2::3]
    
    # 转换为一维列表
    x_coords = [item for sublist in x_coords for item in sublist]
    y_coords = [item for sublist in y_coords for item in sublist]
    z_coords = [item for sublist in z_coords for item in sublist]
    
    return x_coords, y_coords, z_coords

def plot_coordinates_from_two_csv(file_path1, file_path2):
    # 读取第一个 CSV 文件
    x1, y1, z1 = read_coordinates_from_csv(file_path1)
    # 读取第二个 CSV 文件
    x2, y2, z2 = read_coordinates_from_csv(file_path2)
    
    # x3, y3, z3 = read_coordinates_from_csv(file_path3)
    # 创建三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制第一个文件的坐标，颜色为蓝色
    ax.plot(x1, y1, z1, marker='o', color='b', label='File 1', linewidth=0.1, markersize= 1)
    
    # 绘制第二个文件的坐标，颜色为红色
    ax.plot(x2, y2, z2, marker='^', color='r', label='File 2', linewidth=0.1, markersize= 1)
    
    # ax.plot(x3, y3, z3, marker='X', color='G', label='File 3', linewidth=0.1, markersize= 1)
    # 设置轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 添加图例
    ax.legend()
    
    # 显示图表
    plt.show()

# 调用函数并传入两个 CSV 文件路径
# plot_coordinates_from_two_csv('data/traject/screw/capsule_pos.csv', 'data/traject/screw/target_pos.csv', 'data/traject/screw/capsule_train_pos.csv')

# plot_coordinates_from_two_csv('data/traject/square/capsule_pos.csv', 'data/traject/square/target_pos.csv', 'data/traject/square/capsule_train_pos.csv')
plot_coordinates_from_two_csv('data/traject/screw/capsule_pos.csv', 'data/traject/screw/target_pos.csv')
"""查看loguniform分布"""
# import torch
# import matplotlib.pyplot as plt

# # 定义范围和样本数量
# low = 0.3
# high = 3.0
# num_samples = 1000

# # 生成 uniform 分布的样本
# uniform_samples = torch.empty(num_samples).uniform_(low, high).numpy()

# # 生成 loguniform 分布的样本
# loguniform_samples = torch.exp(torch.empty(num_samples).uniform_(torch.log(torch.tensor(low)), torch.log(torch.tensor(high)))).numpy()

# # 绘制 uniform 分布直方图
# plt.hist(uniform_samples, bins=50, density=True, alpha=0.6, color='b', label='Uniform')
# plt.title('Uniform vs Loguniform Distribution')
# plt.xlabel('Value')
# plt.ylabel('Density')

# # 绘制 loguniform 分布直方图
# plt.hist(loguniform_samples, bins=50, density=True, alpha=0.6, color='g', label='Loguniform')
# plt.legend()
# plt.show()