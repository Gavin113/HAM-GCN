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
                    (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24,14), #mouth
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
                    ]   # 原始gragh

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

hand_32connect=[    (25,26), (26, 27), (27, 28), (28, 29),   #left_hand
                    (25,30), (30, 31), (31, 32), (32, 33), 
                    (25,34), (34,35), (35, 36), (36, 37),
                    (25,38), (38,39), (39, 40), (40, 41),
                    (25,32), (42,43), (43, 44), (44, 45), 
                    (27,31), (28,32), (29,33),    #左手关节层级连接
                    (31,35), (32,36), (33,37),
                    (35,39), (36,40), (37,41),
                    (39,43), (40,44), (41,45),
                    (46,47), (47, 48), (48, 49), (49, 50),  #right_hand
                    (46,51), (51, 52), (52, 53), (53, 54),
                    (46,55), (55, 56), (56, 57), (57, 58), 
                    (46,59), (59, 60), (60, 61), (61, 62), 
                    (46,63), (63, 64), (64, 65), (65, 66),
                    (48,52), (49,53),(50,54),    #右手关节层级连接
                    (52,56), (53,57),(54,58),
                    (56,60), (57,61),(58,62),
                    (60,64), (61,65),(62,66),
                    (25,46), #左右手的根节点连接起来
                    ]

# 共32个连接，论文：《Hand Graph Topology Selection for Skeleton-based Sign Language Recognition》
hand_32connect = [(i - 1, j - 1) for (i, j) in hand_32connect]
# print(len(hand_32connect))

hand_24connect = [(25, 26), (26, 27), (27, 28), (28, 29),  # left hand
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
hand_24connect = [(i - 1, j - 1) for (i, j) in hand_24connect]


hand_edge = [(25, 26), (26, 27), (27, 28), (28, 29),   #left_hand   手部骨架连接
                    (30, 31), (31, 32), (32, 33), 
                    (34, 35), (35, 36), (36, 37),
                    (38, 39), (39, 40), (40, 41),
                    (42, 43), (43, 44), (44, 45), 
                    (46, 47), (47, 48), (48, 49), (49, 50),  #right_hand
                    (51, 52), (52, 53), (53, 54),
                    (55, 56), (56, 57), (57, 58), 
                    (59, 60), (60, 61), (61, 62), 
                    (63, 64), (64, 65), (65, 66)
                    ]
hand_edge = [(i - 1, j - 1) for (i, j) in hand_edge]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

hand_layer = [(26, 30), (26, 34), (26, 38), (26, 42), (30, 34), (30, 38), (30, 42), (34, 38), (34, 42), (38, 42), (27, 31), (27, 35), (27, 39), (27, 43), (31, 35), (31, 39), (31, 43), (35, 39), (35, 43), (39, 43), (28, 32), (28, 36), (28, 40), (28, 44), (32, 36), (32, 40), (32, 44), (36, 40), (36, 44), (40, 44), (29, 33), (29, 37), (29, 41), (29, 45), (33, 37), (33, 41), (33, 45), (37, 41), (37, 45), (41, 45), (47, 51), (47, 55), (47, 59), (47, 63), (51, 55), (51, 59), (51, 63), (55, 59), (55, 63), (59, 63), (48, 52), (48, 56), (48, 60), (48, 64), (52, 56), (52, 60), (52, 64), (56, 60), (56, 64), (60, 64), (49, 53), (49, 57), (49, 61), (49, 65), (53, 57), (53, 61), (53, 65), (57, 61), (57, 65), (61, 65), (50, 54), (50, 58), (50, 62), (50, 66), (54, 58), (54, 62), (54, 66), (58, 62), (58, 66), (62, 66)]
hand_layer = [(i - 1, j - 1) for (i, j) in hand_layer]   # 将手部的关节点分层，寻找对应关系


opposite_handindex = [(25, 46), (26, 47), (27, 48), (28, 49), (29, 50), (30, 51), (31, 52), (32, 53), (33, 54), (34, 55), (35, 56), (36, 57), (37, 58), (38, 59), (39, 60), (40, 61), (41, 62), (42, 63), (43, 64), (44, 65), (45, 66)]
opposite_handindex =  [(i - 1, j - 1) for (i, j) in opposite_handindex]  # 将左右手对应的关节点连接

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
            A = tools.get_spatial_hand_graph(num_node, self_link, inward, outward, hand_index)
        elif labeling_mode == 'hand32':
            A = tools.get_32connect_hand_graph(num_node, self_link, inward, outward, hand_32connect)
        elif labeling_mode == 'hand24':
            A = tools.get_32connect_hand_graph(num_node, self_link, inward, outward, hand_24connect)
        elif labeling_mode == 'handedge':
            A = tools.get_handedge_graph(num_node, self_link, inward, outward,hand_edge)
        elif labeling_mode == 'handlayer':
            A = tools.get_handlayer_graph(num_node, self_link, inward, outward,hand_layer)
        elif labeling_mode == 'opphand':
            A = tools.get_handlayer_graph(num_node, self_link, inward, outward,opposite_handindex)
        elif labeling_mode == 'threehand':
            A = tools.get_three_hand_graph(num_node, self_link, inward, outward,hand_index,hand_edge,hand_layer)
        else:
            raise ValueError()
        return A
