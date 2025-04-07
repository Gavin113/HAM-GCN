import torch
import numpy as np
# 根据时间制作mask


hand_index = list(range(24,66))
left_index = hand_index[:21]
right_index = hand_index[21:]

split = 'train'
data_path = f'data/msasl/msasl_{split}.npz'
npz_data = np.load(data_path)
# data = npz_data[f'x_{split}']
data = npz_data['x_test']
y_label = npz_data['y_test']
print(data.shape)
print(y_label)
confidence = data[..., 2]

# 创建mask，置信度大于0.2的为1，否则为0
mask_data = (confidence > 0.2).astype(int)
mask_data = mask_data[:, :, hand_index]
# 统计所有帧的mask总数，得到N,V格式的输出
mask_data = mask_data.sum(axis=2)/42.0

print(mask_data.shape)  # 输出形状应为 (N, V)
print(mask_data)        # 输出 N,V 格式的mask

npz_name = f'data/msasl/msasl_{split}_mask_T.npz' 
print(npz_name,mask_data.shape)
np.savez(npz_name, mask=mask_data)


