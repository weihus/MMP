import torch
import torch.nn as nn
import numpy as np

def calculate_similarity(embedding1, embedding2):
    return nn.functional.cosine_similarity(embedding1, embedding2).mean()

def euclidean_distance(embedding1, embedding2):
    return torch.norm(embedding1 - embedding2)

def aggregate_views(embeddings, weights):
    weighted_sum = torch.zeros_like(embeddings[0])
    for embedding, weight in zip(embeddings, weights):
        weighted_sum += weight * embedding
    return weighted_sum


# def custom_softmax(tensor, distances, suppression_factor, enhancement_factor):
#     # 处理冗余
#     distances = torch.tensor(distances)
#     mask = distances < 5.0
#     new_tensor = tensor.clone()
#     new_tensor[mask] *= 0.
#     # new_tensor = torch.softmax(new_tensor, dim=0)

#     # mask = x_normalized < 0.1
#     # new_tensor[mask]*=0.3
#     # new_tensor = torch.softmax(tensor, dim=0)

#     # 1. 找到最小值
#     min_value, min_index = torch.min(new_tensor, 0)

#     # 2. 判断最小值是否远远偏离其他值，如果是，则将最小值除以2
#     # 这里假设"远远偏离"的定义是最小值小于其他值的均值减去一个标准差
#     mean_value = torch.mean(new_tensor)
#     std_dev = torch.std(new_tensor)
#     if min_value < (mean_value - std_dev):
#         new_tensor[min_index] *= suppression_factor
#     # disney 1.2

#     # 3. 找到远远大于其他值的最大值
#     max_value, max_index = torch.max(new_tensor, 0)


#     # disney 1.1
#     # 4. 将最大值放大2倍
#     # mask2 = x_normalized > 0.1
#     # new_tensor[mask]*=enhancement_factor
#     new_tensor[max_index] = new_tensor[max_index] * enhancement_factor

#     new_tensor = torch.softmax(new_tensor, dim=0)

#     return new_tensor
def adjust(num, num_mean):
    gamma = 10 # books 0, disney  1  enron 10 cora 1
    res = 1.0 / (1+torch.exp(-gamma*(num-num_mean)))
    return res
# def adjust(weight, num_mean):
#     beta = 0.1
#     alpha = 0.1
#     if weight > num_mean:
#         return 1.0 + beta*(weight - num_mean)
#     else:
#         return 1.0 - alpha*(num_mean - weight)

def custom_softmax(tensor, distances, suppression_factor, enhancement_factor):
    # tensor = torch.softmax(tensor, dim=0)

    new_tensor = tensor.clone()
    num_mean = torch.mean(tensor)
    # for idx, i in enumerate(tensor):
    #     new_tensor[idx] = adjust(i, num_mean)
    new_tensor = torch.softmax(new_tensor, dim=0)
    # new_tensor = torch.ones_like(tensor)
    return new_tensor

# def custom_softmax(tensor, distances, suppression_factor, enhancement_factor):
#     distances = torch.tensor(distances)
#     new_tensor = tensor.clone()
#     # sum_x = torch.sum(torch.abs(distances))
#     # x_normalized = distances / sum_x
#     # mask = x_normalized < 0.1
#     # new_tensor[mask] *= new_tensor[mask] * 0.9
#     # for idx, i in enumerate(mask):
#     #     if i and new_tensor[idx]>0.9:
#     #         new_tensor[idx] *= 0.9
#     # new_tensor[mask] *= 0.9
#     # mask2 = new_tensor > 0.7

#     # new_tensor[mask2] *= enhancement_factor

#     # 1. 找到最小值
#     min_value, min_index = torch.min(new_tensor, 0)

#     # 2. 判断最小值是否远远偏离其他值，如果是，则将最小值除以2
#     # 这里假设"远远偏离"的定义是最小值小于其他值的均值减去一个标准差
#     mean_value = torch.mean(new_tensor)
#     std_dev = torch.std(new_tensor)
#     if min_value < (mean_value - std_dev):
#         new_tensor[min_index] *= suppression_factor
#     # disney 1.2

#     # 3. 找到远远大于其他值的最大值
#     max_value, max_index = torch.max(new_tensor, 0)


#     # disney 1.1
#     # 4. 将最大值放大2倍
#     new_tensor[max_index]  *= enhancement_factor
#     # new_tensor[min_index] = new_tensor[min_index] * 0

#     # print(new_tensor)
#     # 5. 找到相似的值并给出它们的索引
#     # 这里假设"相似"的定义是差值在0.05以内
#     # similarity_threshold = 0.05
#     # similar_indices = []
#     # for i, value in enumerate(new_tensor):
#     #     similar_group = (torch.abs(new_tensor - value) < similarity_threshold).nonzero(as_tuple=True)[0]
#     #     if len(similar_group) > 1:
#     #         similar_indices.append(similar_group.tolist())

#     # # 去重并展平相似索引列表
#     # similar_indices = list(set([item for sublist in similar_indices for item in sublist]))
#     # # print(similar_indices)
#     # # raise
#     # # 5. 相似值的权重乘以欧氏距离
#     # distances = torch.tensor(distances)
#     # new_distances = distances[similar_indices]
#     # new_distances = new_distances / torch.sum(new_distances) *10

#     # for idx, similar_index in enumerate(similar_indices):
#     #     new_tensor[similar_index] *= new_distances[idx]
#     new_tensor = torch.softmax(new_tensor, dim=0)
#     # new_tensor[0] = 0
#     # new_tensor[1] = 0
#     # new_tensor[2] = 0
#     # new_tensor[4] = 0
#     return new_tensor


def find_top_k_indices(arr, k=10):
    # 使用numpy.argsort获取排序后的索引
    sorted_indices = np.argsort(arr)
    # 获取前k个最大值的索引（因为argsort返回的是从小到大排序的索引，所以我们需要从末尾开始取）
    top_k_indices = sorted_indices[-k:]
    # 由于是从末尾开始取，所以索引是逆序的，我们再反转一下
    return top_k_indices[::-1]
