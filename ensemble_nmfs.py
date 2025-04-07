import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        default='nmf',
                        choices={'ntu/xsub','nmf','nmf_normal','nmf_comfusing'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=[0.6, 0.5, 0.6],
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        default='work_dir/nmfs/joint_gcn_mask4_80.49',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        default='work_dir/nmfs/bone_gcn_mask-79.24',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', 
                        default='work_dir/nmfs/joint_motion_gcn_mask4-77.75',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-motion-dir', 
                        default= 'work_dir/nmfs/bone_motion_gcn_mask-74.86')
    return parser

def softmax(x):
    exp_x = np.exp(x)  
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def get_accuracy(r1=None,r2=None,r3=None,r4=None,label=None,alpha = [0.5,0.5,0.5,0.5],if_softmax=True):
    right_num_5 = 0
    right_num_2 = 0
    right_num = 0
    total_num = 0

    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]            
        _, r33 = r3[i]
        _, r44 = r4[i]
        if if_softmax:
            r11 = softmax(r11)
            r22 = softmax(r22)
            r33 = softmax(r33)
            r44 = softmax(r44)
        r = r11 * alpha[0] + r22 *alpha[1] + r33 * alpha[2] + r44 * alpha[3]
        rank_2 = r.argsort()[-2:]
        right_num_2 += int(int(l) in rank_2)
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc2 = right_num_2 / total_num
    acc5 = right_num_5 / total_num
    print(right_num,right_num_5,total_num)
    return acc,acc2,acc5
    
if __name__ == "__main__":
    parser = get_parser()
    
    arg = parser.parse_args()
    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'nmf' in arg.dataset:
        npz_data = np.load('data/NMFs-CSL/nmf_test.npz')
        label = npz_data['y_test']
        if arg.dataset == 'nmf_normal':
            ind = np.where(label>609)
            label = label[ind] - 610
        elif arg.dataset == 'nmf_comfusing':
            ind = np.where(label<610)
            label = label[ind]
            
    else:
        raise NotImplementedError
    
    test_score_pkl = 'epoch1_test_score.pkl'
    with open(os.path.join(arg.joint_dir, test_score_pkl), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, test_score_pkl), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, test_score_pkl), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, test_score_pkl), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    right_num = total_num = right_num_5 = 0
    alpha_list=[[0.5,0.5,0.5,0.5],   
                ]
    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        for alpha in alpha_list:
            
            acc,acc2,acc5=get_accuracy(r1,r2,r3,r4,label,alpha)
            print(alpha)
            print('Top1 Acc: {:.4f}%'.format(acc * 100))
            print('Top2 Acc: {:.4f}%'.format(acc2 * 100))
            print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
            print('Right_num:{} Total_num:{}'.format(right_num,total_num))

    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        arg.alpha = [0.6, 0.5, 0.6]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        
    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 + r22 * arg.alpha
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        
    
