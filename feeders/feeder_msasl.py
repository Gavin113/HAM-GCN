import numpy as np

from torch.utils.data import Dataset
import pickle
import json
from feeders import tools
pose_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1,num_class = 1000, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.num_class = num_class
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C T V M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train']
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = npz_data['y_test']
        else:
            raise NotImplementedError('data split only supports train/test')
        if self.num_class == 100:
            self.ind = np.where(self.label<100)
            self.label = self.label[self.ind] 
            self.data = self.data[self.ind]
        elif self.num_class == 200:
            self.ind = np.where(self.label<200)
            self.label = self.label[self.ind]
            self.data = self.data[self.ind]
        elif self.num_class == 500:
            self.ind = np.where(self.label<500)
            self.label = self.label[self.ind]
            self.data = self.data[self.ind]
        self.sample_name = ['data_' + str(i) for i in range(len(self.data))]
        # self.data = self.data[:,:,pose_index,:]
        N, T, V, C = self.data.shape
        
        self.data = self.data.reshape((N, T, V, C ,1 )).transpose(0, 3, 1, 2, 4)
        
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
        print(self.split,self.data.shape)  # N,C,T,V,M

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        # self.data = (self.data - self.mean_map) / self.std_map
        # print(self.data.shape,'data normalization done')

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot and np.random.rand() < 0.3:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Feeder2(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1,num_class =1000, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.num_class = num_class
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        # if normalization:
        #     self.get_mean_map()

    def load_data(self):
        # data: N C T V        
        pkl_file = "data/MSASL/msasl/keypoints_hrnet_dark_coco_wholebody.pkl"
        with open(pkl_file, 'rb') as file:
            dataset = pickle.load(file)
            
        msasl_classes = json.load(open("data/MSASL/msasl/MSASL_classes.json"))
        num_class = int(self.num_class)
        msasl_classes = msasl_classes[:num_class]
        self.label_dict = {label: i for i, label in enumerate(msasl_classes)}
        if self.split == 'train':
            split_file =  "data/MSASL/msasl/train.pkl"
        elif self.split == 'test':
            split_file =  "data/MSASL/msasl/test.pkl"
        elif self.split == 'val':
            split_file =  "data/MSASL/msasl/dev.pkl"
        with open(split_file, 'rb') as file:
            splitdata = pickle.load(file)
            
        splitdata = [item for item in splitdata if item['label'] in msasl_classes]
        
        self.names = [item['name'] for item in splitdata]
        self.label = [item['label'] for item in splitdata]
        self.data = {key: value for key, value in dataset.items() if key in self.names}
        self.sample_name = [self.names[i] +'_'+ str(i) for i in range(len(self.data))]
        print(self.split,':\t', len(self.names))
        print(self.split,':\t', len(self.data))
        


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_name = self.names[index]
        data_numpy = self.data[data_name]
        label = self.label[index]
        label = int(self.label_dict[label])
        data_numpy = np.array(data_numpy)
        data_numpy = data_numpy[:,pose_index,:]    # 取66个关键点
        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            data_numpy = data_numpy / [140.0,140.0,1.0]
        # reshape T, V, C 到 C, T, V
        data_numpy = data_numpy.transpose(2, 0, 1) 
        # 统计有效帧
      
        # confidence = data_numpy[2, :, [26, 47]]  # 提取置信度通道 (C=2)，仅保留索引 26 和 47 的关键点，形状为 (T, 2)
        confidence = data_numpy[2, :, : ]
        valid_mask = (confidence > 0.1)  #  判断置信度 > 0.2 的帧, 布尔数组，形状为 (T, 2)     
        valid_frames = valid_mask.any(axis=0) 
        valid_frame_num =  np.sum(valid_frames) #统计至少一个关键点置信度 > 0.2 的有效帧数,对时间帧 (T) 统计
        valid_frame_num = valid_frame_num if valid_frame_num > 30 else data_numpy.shape[1]
        
        # valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize2(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


class Feeder3(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1,num_class = 1000, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.num_class = num_class
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C T V M
        npz_data = np.load(self.data_path)
        self.data = npz_data['x_data']
        self.label = npz_data['y_label']
        self.sample_name = npz_data['name']
            
        # self.data = self.data[:,:,pose_index,:]
        N, T, V, C = self.data.shape
        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            self.data = self.data / [140.0,140.0,1.0]
        
        self.data = self.data.reshape((N, T, V, C ,1 )).transpose(0, 3, 1, 2, 4)
        print(self.split,self.data.shape)  # N,C,T,V,M

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        # self.data = (self.data - self.mean_map) / self.std_map
        # print(self.data.shape,'data normalization done')

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

