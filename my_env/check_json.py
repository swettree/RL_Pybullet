import torch

checkpoint = torch.load('UR10eMagStack.pth')
state_dict = checkpoint['model']

# 打印每个键对应 value 的形状
for key, value in state_dict.items():
    print(f'{key}: {value.shape}')