import torch
import numpy as np
import scipy.sparse as sp
from pygod.utils import load_data
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import NormalizeFeatures

transform = NormalizeFeatures()


def split_view(data_original, view_list):
    data_list = []

    for view in view_list:
        view_x = data_original.x[:, view]
        data_view = Data(x=view_x, edge_index=data_original.edge_index, y=data_original.y)
        data_view = transform(data_view)
        data_list.append(data_view)
    return data_list

def process_disney(data_original):
    # keys = ["MinPriceUsedItem", "Rating_3_Ratio", "Rating_of_review_with_least_votes", "Helpful_votes_ratio", "Rating_4_Ratio", "Amazon_price", 
    #     "Number_of_reviews", "Review_frequency", "Min_Votes", "Avg_Votes", "Rating_of_least_helpful_rating", "MinPricePrivateSeller", "Rating_1_Ratio", 
    #     "Max_Votes", "Rating_5_Ratio", "Number_of_different_authors", "Rating_of_review_with_most_votes", "Min_Helpful", "Rating_span", "Sales_Rank",
    #     "No_of_Categories", "Max_Helpful", "Product_group", "Avg_Rating", "Avg_Helpful", "Rating_of_most_helpful_rating", "Top_reviewer_rating", "Rating_2_Ratio"]
    ratingInfo = [1, 4, 12, 14, 27, 23, 25, 26, 10, 16, 2, 18]
    priceInfo = [0, 5, 11]
    reviewInfo = [6, 7, 15, 21, 24, 3, 8, 9, 13, 17]
    other_info = [19, 20, 22]

    # last = ratingInfo + priceInfo + reviewInfo + other_info
    # view_list = [last, ratingInfo, priceInfo, reviewInfo, other_info]

    last = ratingInfo + priceInfo + reviewInfo + other_info
    view_list = [last, ratingInfo, priceInfo, reviewInfo, other_info]

    data_list = split_view(data_original, view_list)
    return data_list

def process_books(data_original):
    # https://dbis.ipd.kit.edu/mitarbeiter/muellere/consub/RealData/AttributesAmazon.txt
    # keys = ["Min_Votes", "Rating_3_Ratio", "MinPriceUsedItem", "Rating_4_Ratio", "Helpful_votes_ratio", "Number_of_reviews",
    #         "Amazon_price", "Review_frequency", "Rating_span", "Sales_Rank", "Min_Helpful", "Max_Helpful", "Avg_Votes", "MinPricePrivateSeller",
    #         "Rating_1_Ratio", "rnmlcCluster", "Avg_Helpful", "Avg_Rating", "Max_Votes", "Rating_5_Ratio", "Rating_2_Ratio"]
    priceInfo = [13, 2, 6]
    ratingInfo = [1, 3, 8, 14, 17, 19, 20]
    reviewInfo = [0, 4, 5, 7, 10, 11, 12, 16, 18]
    other_info = [9, 15]
    test_info = [15]
    # ratingInfo.sort()
    # last = ratingInfo + priceInfo  + reviewInfo + other_info + test_info
    # view_list = [last, ratingInfo, priceInfo, reviewInfo, other_info, test_info]

    # alarm
    last = ratingInfo + priceInfo  + reviewInfo + other_info# + test_info
    view_list = [last, ratingInfo, priceInfo, reviewInfo, other_info]#, test_info]

    data_list = split_view(data_original, view_list)
    return data_list


def process_enron(data_original):
    # https://dbis.ipd.kit.edu/mitarbeiter/muellere/consub/RealData/AttributesEnron.txt
    # https://dbis.ipd.kit.edu/mitarbeiter/muellere/consub/
    keys = ['AverageContentForwardingCount', 'OtherMailsTo', 'MimeVersionsCount', 'AverageRangeBetween2Mails', 'AverageDifferentSymbolsContent', 
            'AverageDifferentSymbolsSubject', 'DifferentEncodingsCount', 'AverageNumberTo', 'DifferntCosCount', 'OtherMailsBcc', 'EnronMailsBcc', 
            'OtherMailsCc', 'EnronMailsTo', 'DifferentCharsetsCount', 'AverageContentReplyCount', 'AverageNumberCc', 'AverageContentLength', 'AverageNumberBcc']

    recSendInfo = [12, 1, 7, 11, 9, 10, 17, 15]
    cntInfo = [16, 14, 0, 4, 6, 13, 2, 8]
    titleRateInfo = [3, 5]

    # cntInfo.sort()
    # last = cntInfo + recSendInfo + titleRateInfo
    # view_list = [last, cntInfo, recSendInfo, titleRateInfo]
    
    # alarm
    last = cntInfo + recSendInfo + titleRateInfo
    view_list = [last, cntInfo, recSendInfo, titleRateInfo]

    data_list = split_view(data_original, view_list)

    return data_list
def process_single_view(dataname):
    if dataname == 'inj_cora':
        views = 2
    elif dataname == "weibo":
        views = 2
    elif dataname == "inj_amazon" or dataname == "reddit":
        views = 3
    # views = 2
    data = load_data(dataname)
    data.y = data.y.bool()
    features_num = data.x.shape[1]

    # 计算每组应该有多少个元素
    group_size = features_num // views  # 整除得到每组基本数量
    remainder = features_num % views  # 取模得到剩余的数量

    # 初始化结果列表
    view_list = []

    # 分配每组的元素
    start_index = 0
    for i in range(views):
        # 如果还有余数，将余数分配给前几组
        end_index = start_index + group_size + (1 if i < remainder else 0)
        # 添加当前组的索引到结果列表
        view_list.append(list(range(start_index, end_index)))
        # 更新起始索引
        start_index = end_index

    views = []
    
    for view in view_list:
        view_x = data.x[:, view]
        data_view = Data(x=view_x, edge_index=data.edge_index, y=data.y)
        if dataname != "weibo":
            data_view = transform(data_view)
        # data_view = transform(data_view)
        views.append(data_view)
    return views
    # edge_index = data["edge_index"]
    # adj_matrix = to_dense_adj(edge_index)
    # adj = adj_matrix.squeeze(0)
    # adj = adj+torch.eye(adj.shape[0])

    # # raise
    # adj_norm = normalize_adj(adj)
    # adj_norm = adj_norm.toarray()

    # feat = data.x
    # label = data.y

    # feats = []
    # for view in views:
    #     ft = view.x
   
    #     feats.append(ft)

    # return adj_norm, feat, feats, label, adj, views, edge_index
def load_dataset(dataset):

    original_data = load_data(dataset)
    original_data.y = original_data.y.bool()
    
    if dataset == "disney":
        views = process_disney(original_data)
        edge_index = original_data.edge_index
        original_data = views[0]
        views = views[1:]

    elif dataset == "books":
        views = process_books(original_data)
        edge_index = original_data.edge_index
        original_data = views[0]
        views = views[1:]
        
    elif dataset == "enron":
        views = process_enron(original_data)
        edge_index = original_data.edge_index
        original_data = views[0]
        views = views[1:]
    else:
        views = process_single_view(dataset)
        edge_index = original_data['edge_index']


    # original_data = transform(original_data)
    # if dataset == 'weibo':
    #     original_data = transform(original_data)

    adj_matrix = to_dense_adj(edge_index)
    adj = adj_matrix.squeeze(0)
    adj = adj+torch.eye(adj.shape[0])

    # raise
    adj_norm = normalize_adj(adj)
    adj_norm = adj_norm.toarray()

    feat = original_data.x
    label = original_data.y

    feats = []
    for view in views:
        # view = transform(view)
        ft = view.x
   
        feats.append(ft)
    return adj_norm, feat, feats, label, adj, views, edge_index

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
