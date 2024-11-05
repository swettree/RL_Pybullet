import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
file_path = 'output_4.csv'
data = pd.read_csv(file_path, header=None)

# 获取总误差列
total_error = data.iloc[:, 3].values*1000

# 获取最小值和最大值
min_error = total_error.min()
max_error = total_error.max()

mean_error = total_error.mean()
print(mean_error)
# 假设 x, y, z 的步长是 0.02m, 覆盖的空间是 200x200x200mm
x_steps = np.arange(0, 0.22, 0.02)
y_steps = np.arange(0, 0.22, 0.02)
z_steps = np.arange(0, 0.22, 0.02)


# 选取 z = 0，40，80，120，160，200 对应的索引
selected_indices = [0, 2, 4, 6, 8, 10]

# 按照 z -> y -> x 的顺序重新排列数据
error_grid = total_error.reshape((len(z_steps), len(y_steps), len(x_steps)))

# 调整维度顺序为 x -> y -> z
error_grid = error_grid.transpose(2, 1, 0)

# 对选定的 z-y 平面（在 z 不同的情况下）绘制热点图
fig, axs = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)
axs = axs.flatten()

for idx, i in enumerate(selected_indices):
    ax = axs[idx]
    # 使用 min_error 和 max_error 作为颜色映射范围
    cax = ax.imshow(error_grid[:, :, i], cmap='Reds', origin='lower', extent=[0, 200, 0, 200], vmin=min_error, vmax=max_error)
    ax.set_title(f'z = {z_steps[i]*1000:.1f} mm')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    fig.colorbar(cax, ax=ax, orientation='vertical')

plt.suptitle('Error Heatmaps Across Selected Z Levels')
plt.show()



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 读取数据
# file_path = 'output_4.csv'
# data = pd.read_csv(file_path, header=None)

# # 获取总误差列
# total_error = data.iloc[:, 3].values

# # 假设 x, y, z 的步长是 0.02m, 覆盖的空间是 200x200x200mm
# x_steps = np.arange(0, 0.22, 0.02)
# y_steps = np.arange(0, 0.22, 0.02)
# z_steps = np.arange(0, 0.22, 0.02)

# # 按照 z -> y -> x 的顺序重新排列数据
# error_grid = total_error.reshape((len(z_steps), len(y_steps), len(x_steps)))

# # 调整维度顺序为 x -> y -> z
# error_grid = error_grid.transpose(2, 1, 0)

# # 创建 3D 网格
# X, Y, Z = np.meshgrid(x_steps * 1000, y_steps * 1000, z_steps * 1000, indexing='ij')

# # 将数据展平，以便用于 3D 散点图
# X_flat = X.flatten()
# Y_flat = Y.flatten()
# Z_flat = Z.flatten()
# errors_flat = error_grid.flatten()

# # 绘制 3D 散点图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 使用颜色表示误差大小，颜色越深表示误差越大
# scat = ax.scatter(X_flat, Y_flat, Z_flat, c=errors_flat, cmap='Reds', marker='o')

# # 添加颜色条
# cbar = plt.colorbar(scat, ax=ax, shrink=0.5, aspect=5)
# cbar.set_label('Error')

# # 设置标签
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.set_title('3D Error Distribution')

# plt.show()
