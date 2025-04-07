
import torch
import numpy as np
class class_accuracy:
    def __init__(self, class_num, topk=[1,2,5,10]):
        self.num = class_num  # 类别数
        self.topk = topk  # 需要计算的top-k值，接受一个列表
        self.correct_list = {k: [0 for _ in range(self.num)] for k in self.topk}  # 每个topk的正确分类次数
        self.list = [0 for _ in range(self.num)]  # 每个类别的总样本数

    def update(self, output, target):
        """更新每个类别的正确分类次数"""
        with torch.no_grad():
            batch_size = target.size(0)  # 获取当前batch的大小

            # 获取每个样本的top-k预测
            _, pred = output.topk(max(self.topk), 1, True, True)  # (batch_size, max(topk))

            # 遍历每个样本并更新正确分类计数
            for i in range(batch_size):
                self.list[target[i]] += 1  # 更新类别的总样本数

                for k in self.topk:
                    if target[i] in pred[i, :k]:  # 如果目标类别出现在top-k预测中
                        self.correct_list[k][target[i]] += 1  # 更新该类别的top-k正确分类次数
    def compute_avg(self):
        """计算每个top-k的平均准确率"""
        correct_list_topk = {k: np.array(v, dtype=np.float32) for k, v in self.correct_list.items()}
        self.list = np.array(self.list, dtype=np.float32)

        avg_acc = {}
        for k in self.topk:
            avg_acc[k] = np.mean(correct_list_topk[k] / self.list)  # 计算每个top-k的平均准确率

        return avg_acc
                        
                        
def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2