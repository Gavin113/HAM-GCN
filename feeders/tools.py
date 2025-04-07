import random
import matplotlib.pyplot as plt
import numpy as np
import pdb

import torch
import torch.nn.functional as F

def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    if len(data_numpy.shape) ==4:
        C, T, V, M = data_numpy.shape
    elif len(data_numpy.shape) ==3:
        C, T, V = data_numpy.shape
        M = 1
        data_numpy = np.expand_dims(data_numpy, axis=3)
    
    if valid_frame_num > 64:
        # 随机选择一个起始点，使得范围是 [begin, begin + 64]
        begin = np.random.randint(0, valid_frame_num - 64 + 1)  # 范围为 [0, valid_frame_num - 64]
        end = begin + 64
    else:
        # 如果帧数不足 64，直接使用全部帧
        begin = 0
        end = valid_frame_num

    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size) # constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data_lenth = data.shape[1]
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data

def valid_crop_resize2(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    if len(data_numpy.shape) ==4:
        C, T, V, M = data_numpy.shape
    elif len(data_numpy.shape) ==3:
        C, T, V = data_numpy.shape
        M = 1
        data_numpy = np.expand_dims(data_numpy, axis=3)
    
    if valid_frame_num >= 64:
        # 随机选择一个起始点，使得范围是 [begin, begin + 64]
        begin = np.random.randint(0, valid_frame_num - 64 + 1)  # 范围为 [0, valid_frame_num - 64]
        end = begin + 64
    else:
        # 如果帧数不足 64，直接使用全部帧
        begin = 0
        end = valid_frame_num

    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size) # constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data_lenth = data.shape[1]
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, data_lenth)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data
def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size = 64, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def random_choose_AUTSL(data_numpy,mask,size = 64):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy,mask
    elif T > size:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :], mask[begin:begin + size]



def random_choose2(data_numpy, size = 50, auto_pad=True):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        # 随机选择一段长度为 size 的片段
        start = random.randint(0, T - size)
        output = np.tile(data_numpy[:, 0:1, :, :], (1, T, 1, 1))  # 将第0帧复制T次
        output[:, start:start + size, :, :] = data_numpy[:, start:start + size, :, :]
        return output


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3):
    """
    对数据进行随机旋转操作
    data_numpy: C,T,V
    """
    data_torch = torch.from_numpy(data_numpy)
    
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()
    data_numpy = np.array(data_torch)

    return data_numpy

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy

def get_selected_indexs(vlen, num_frames=64, is_train=True, setting=['consecutive', 'pad', 'central', 'pad']):
    pad = None  #pad denotes the number of padding frames
    assert len(setting) == 4
    # denote train > 64, test > 64, test < 64
    train_p, train_m, test_p, test_m = setting
    assert train_p in ['consecutive', 'random']
    assert train_m in ['pad']
    assert test_p in ['central', 'start', 'end']
    assert test_m in ['pad', 'start_pad', 'end_pad']
    if num_frames > 0:
        assert num_frames%4 == 0
        if is_train:
            if vlen > num_frames:
                if train_p == 'consecutive':
                    start = np.random.randint(0, vlen - num_frames, 1)[0]
                    selected_index = np.arange(start, start+num_frames)
                elif train_p == 'random':
                    # random sampling
                    selected_index = np.arange(vlen)
                    np.random.shuffle(selected_index)
                    selected_index = selected_index[:num_frames]  #to make the length equal to that of no drop
                    selected_index = sorted(selected_index)
                else:
                    selected_index = np.arange(0, vlen)
            elif vlen < num_frames:
                if train_m == 'pad':
                    remain = num_frames - vlen
                    selected_index = np.arange(0, vlen)
                    pad_left = np.random.randint(0, remain, 1)[0]
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
            else:
                selected_index = np.arange(0, vlen)
        
        else:
            if vlen >= num_frames:
                start = 0
                if test_p == 'central':
                    start = (vlen - num_frames) // 2
                elif test_p == 'start':
                    start = 0
                elif test_p == 'end':
                    start = vlen - num_frames
                selected_index = np.arange(start, start+num_frames)
            else:
                remain = num_frames - vlen
                selected_index = np.arange(0, vlen)
                if test_m == 'pad':
                    pad_left = remain // 2
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'start_pad':
                    pad_left = 0
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'end_pad':
                    pad_left = remain
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
    else:
        # for statistics
        selected_index = np.arange(vlen)

    return selected_index, pad

def pad_array(array, l_and_r):
    left, right = l_and_r
    if left > 0:
        pad_img = array[0]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([left]+[1]*(len(array.shape)-1)))
        array = np.concatenate([pad, array], axis=0)
    if right > 0:
        pad_img = array[-1]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([right]+[1]*(len(array.shape)-1)))
        array = np.concatenate([array, pad], axis=0)
    return array

