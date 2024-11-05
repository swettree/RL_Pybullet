import torch
import json

# 加载 pth 文件
pth_file_path = 'UR10eMagStack.pth'
model_weights = torch.load(pth_file_path)






# 获取每个键的形状，并将数据转换为可序列化的格式
weights_shapes = {}
serializable_weights = {}

def convert_to_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()  # 将 tensor 转换为列表
    elif isinstance(value, (int, float, str, bool)):
        return value  # 这些类型是天然可序列化的
    elif isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()}  # 递归处理字典
    elif isinstance(value, (list, tuple)):
        return [convert_to_serializable(v) for v in value]  # 递归处理列表和元组
    else:
        return str(value)  # 最后尝试将其转换为字符串

for key, value in model_weights.items():
    try:
        weights_shapes[key] = list(value.shape)
    except AttributeError:
        weights_shapes[key] = str(type(value))
    serializable_weights[key] = convert_to_serializable(value)

# 打印每个键的形状
for key, shape in weights_shapes.items():
    print(f'{key}: {shape}')

# 将所有内容保存到 json 文件中
json_file_path = 'model_weights.json'
with open(json_file_path, 'w') as json_file:
    json.dump(serializable_weights, json_file)

print(f'All model weights have been saved to {json_file_path}')

