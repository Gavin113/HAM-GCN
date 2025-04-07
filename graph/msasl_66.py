import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

Hrnet_Part2index = {
    'upbody':list(range(13)),
    'pose': list(range(11)),
    'hand': list(range(91, 133)),
    'mouth': list(range(71,82)),
    'face_others': list(range(23, 71))
}
part_index = ['upbody','mouth', 'hand']
pose_index = []
for k in part_index:
    pose_index += Hrnet_Part2index[k]
# print(pose_index)

num_node = 13 + 11 + 42    # 66
self_link = [(i, i) for i in range(num_node)]

inward_ori_index = [(1,2), (1,3), (2,3), (2,4),(3,5),   #head
                    (5,7), (7,9), (9,11),      # left arm
                    (4,6), (6,8), (8,10),      # right arm
                    (7,13), (12,13),(6,12),
                    (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), #mouth
                    (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24,14),
                    (25, 26), (26, 27), (27, 28), (28, 29),   #left_hand
                    (30, 31), (31, 32), (32, 33), 
                    (34, 35), (35, 36), (36, 37),
                    (38, 39), (39, 40), (40, 41),
                    (42, 43), (43, 44), (44, 45), 
                    (46, 47), (47, 48), (48, 49), (49, 50),  #right_hand
                    (51, 52), (52, 53), (53, 54),
                    (55, 56), (56, 57), (57, 58), 
                    (59, 60), (60, 61), (61, 62), 
                    (63, 64), (64, 65), (65, 66)
                    ] # 原始gragh

inward_ori_index = [ (1, 2), (1, 3),(2, 3), (2, 4), (3, 5),  # head
                    (1,4),(1,5),(1,17),(1,14), #head
                    (6,7),(5, 7), (7, 9), (9, 11),                # left arm
                          (4, 6), (6, 8), (8, 10),                # right arm
                    (7, 13), (12, 13), (6, 12),
                    (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), # mouth
                    (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 14), # mouth
                    (25, 26), (26, 27), (27, 28), (28, 29),  # left hand
                    (25,30),(25,34), (25,38),(25,42),(27,30),(30,34),(34,38),(38,42),  #手指间连接
                    (30, 31), (31, 32), (32, 33),
                    (34, 35), (35, 36), (36, 37),
                    (38, 39), (39, 40), (40, 41),
                    (42, 43), (43, 44), (44, 45),
                    (46, 47), (47, 48), (48, 49), (49, 50),  # right hand
                    (46,51),(46,55),(46,59),(46,63),(48,51),(51,55),(55,59),(59,63), #手指间连接
                    (51, 52), (52, 53), (53, 54),
                    (55, 56), (56, 57), (57, 58),
                    (59, 60), (60, 61), (61, 62),
                    (63, 64), (64, 65), (65, 66)]
hand_index = range(24,66)
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


#infogcn
inward_ori_index_infogcn = [
    (2, 1), (2, 21), (21, 3), (3, 4), #head
    (21, 5), (5, 6), (6, 7), (7, 8), (8, 23), (23, 22), # left arm
    (21, 9), (9, 10), (10, 11), (11, 12), (12, 25), (25, 24), # right arm
    (1, 13), (13, 14), (14, 15),(15, 16), # left leg
    (1, 17), (17, 18),  (18, 19),  (19, 20) # right leg
]
inward_infogcn = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward_infogcn = [(j, i) for (i, j) in inward_infogcn]
neighbor_infogcn = inward_infogcn + outward_infogcn

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.outward_infogcn = outward_infogcn
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'hand':
            A = tools.get_spatial_hand_graph(num_node, self_link, inward, outward,hand_index)
        else:
            raise ValueError()
        return A
