import numpy as np

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def hand_A1(num_node,hand_index): # 全连接
    A = np.zeros((num_node, num_node))
    for i in hand_index:
        for j in hand_index:
            A[i,j]=1
    A_OUT = normalize_digraph(A)
    return A_OUT

def handedge_A(num_node,hand_edge):    # 依据hand的连接数组创建矩阵A
    A = np.zeros((num_node, num_node))
    for index in hand_edge:
        (i,j) = index
        A[i,j]=1
        A[j,i]=1
    A_OUT = normalize_digraph(A)
    return A_OUT



def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_spatial_hand_graph(num_node, self_link, inward, outward,hand_index):
    # 在第4个通道添加手部全连接图
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    Hand = hand_A1(num_node,hand_index)# 全连接
    A = np.stack((I, In, Out,Hand))
    return A

def get_handedge_graph(num_node, self_link, inward, outward,hand_edge):
    # 在第4个通道单独添加手部邻居图
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    Hand = handedge_A(num_node,hand_edge)    # 手部连接点，关节点连接，手部单独的连接图
    A = np.stack((I, In, Out,Hand))
    return A

def get_handlayer_graph(num_node, self_link, inward, outward,hand_index):
    # 在第4个通道单独设置手部层次图
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    Hand = handedge_A(num_node,hand_index)    # 分层次连接，在同一层的全连接
    A = np.stack((I, In, Out,Hand))
    return A

def get_32connect_hand_graph(num_node, self_link, inward, outward, hand_index):
    # 在第4个通道添加手部32连接图
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    Hand = handedge_A(num_node,hand_index)    # 将连接数组转为邻接矩阵A
    A = np.stack((I, In, Out,Hand))
    return A
   
def get_three_hand_graph(num_node, self_link, inward, outward,hand_index,hand_edge,hand_layer):
    # 6个通道，在第3个通道之后添加手部全连接，关节点连接，分层次连接图
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    Hand1 = hand_A1(num_node,hand_index)# 全连接
    Hand2 = handedge_A(num_node,hand_edge)    # 手部连接点，关节点连接，手部单独的连接图
    Hand3 = handedge_A(num_node,hand_layer)    # 分层次连接，在同一层的全连接
    A = np.stack((I, In, Out,Hand1,Hand2,Hand3))
    return A


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A