import torch
import logging
import argparse
import numpy as np
from model import MVGAD
from dataset import load_dataset
from sklearn.metrics import roc_auc_score, f1_score
# import sys
# hid_dim = int(sys.argv[1])
# alpha = float(sys.argv[1])
# beta = float(sys.argv[1])

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='w')
logger = logging.getLogger(__name__)
# datasets = ['inj_cora', 'inj_amazon', 'books', 'reddit', 'weibo', 'disney', 'enron']


# hyperParameter = {
#     "disney":{"hidden_dim": 256, "alpha": 0.6, "beta": 0.1, "lr": 6e-4, "suppression_factor": 0.3, 'enhancement_factor':1.4},
#     "books": {"hidden_dim": 256, "alpha": 0.6, "beta": 0., "lr": 6e-4, "suppression_factor": 0., 'enhancement_factor':1.2}, 
#     # "books": {"hidden_dim": 256, "alpha": 0., "beta": 0., "lr": 4e-5, "suppression_factor": 0.0, 'enhancement_factor':1.1},
#     "enron": {"hidden_dim": 256, "alpha": 0.1, "beta": 0., "lr": 1e-5, "suppression_factor": .5, 'enhancement_factor':2.0},
#     # "enron": {"hidden_dim": 256, "alpha": 0.1, "beta": 0., "lr": 1e-6, "suppression_factor": 0.5, 'enhancement_factor':2.0}, 
#     "inj_cora": {"hidden_dim": 512, "alpha": 0.9, "beta": 0.4, "lr": 1e-7, "suppression_factor": 0., 'enhancement_factor':1.9}, 
#     "inj_amazon": {"hidden_dim": 128, "alpha": 0.99, "beta": 0.1, "lr": 4e-7, "suppression_factor": 0., 'enhancement_factor':1.3},
#     "reddit": {"hidden_dim": 512, "alpha": 0.9, "beta": 0.3, "lr": 1e-6, "suppression_factor": 0., 'enhancement_factor':1.9},
#     "weibo": {"hidden_dim": 64, "alpha": 0.99, "beta": 0., "lr": 1e-6, "suppression_factor": 0., 'enhancement_factor':1.}
# }

hyperParameter = {
    "disney":{"hidden_dim": 256, "alpha": 0.7, "beta": 0.1, "lr": 1e-1, "suppression_factor": 0.3, 'enhancement_factor':1.4}, # 77.01
    # "books": {"hidden_dim": 256, "alpha": 0.8, "beta": 0.001, "lr": 1e-3, "suppression_factor": 0., 'enhancement_factor':1.2},  
    "books": {"hidden_dim": 256, "alpha": 0.8, "beta": 0.1, "lr": 1e-3, "suppression_factor": 0., 'enhancement_factor':1.2}, 
    # "enron": {"hidden_dim": 256, "alpha": 0.01, "beta": 0.001, "lr": 1e-6, "suppression_factor": .5, 'enhancement_factor':2.0},
    "enron": {"hidden_dim": 512, "alpha": 0.1, "beta": 0.3, "lr": 1e-5, "suppression_factor": .5, 'enhancement_factor':2.0},
    "inj_cora": {"hidden_dim": 256, "alpha": 0.9, "beta": 0.7, "lr": 1e-7, "suppression_factor": 0., 'enhancement_factor':1.9}, 
    # "inj_amazon": {"hidden_dim": 64, "alpha": 0.99, "beta": 0.2, "lr": 1e-7, "suppression_factor": 1., 'enhancement_factor':1.1},
    "inj_amazon": {"hidden_dim": 64, "alpha": 0.99, "beta": 0.2, "lr": 1e-7, "suppression_factor": 1., 'enhancement_factor':1.1},
    "reddit": {"hidden_dim": 512, "alpha": 0.5, "beta": 0.1, "lr": 1e-5, "suppression_factor": 0., 'enhancement_factor':1.},
    # "weibo": {"hidden_dim": 256, "alpha": 0.1, "beta": 0.1, "lr": 1e-6, "suppression_factor": 0., 'enhancement_factor':1.},
}




def adaptive_loss_function(reconstructed_attrs, reconstructed_structs, original_attrs, original_structs, original_embedding, aggregated_embedding, alpha, beta):
    diff_attribute = torch.pow(reconstructed_attrs - original_attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attr_loss = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(reconstructed_structs - original_structs, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    struct_loss = torch.mean(structure_reconstruction_errors)

    agg_structure = torch.pow(aggregated_embedding - original_embedding, 2)
    agg_reconstruction_errors = torch.sqrt(torch.sum(agg_structure, 1))
    agg_loss = torch.mean(agg_reconstruction_errors)

    # books
    total_loss = (1-alpha) * attribute_reconstruction_errors + alpha * structure_reconstruction_errors - beta*agg_reconstruction_errors# 综合损失
    # total_loss = beta*agg_reconstruction_errors -  (1-alpha) * attribute_reconstruction_errors - alpha * structure_reconstruction_errors # 综合损失

    return total_loss, attr_loss, struct_loss, agg_loss

def train_mvgad(args):
    res_list = []
    for i in range(10):
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)
        adj, attrs, feats, label, adj_label, views, edge_index = load_dataset(args.dataset)

        # adj = torch.FloatTensor(adj)
        adj_label = torch.FloatTensor(adj_label)
        attrs = torch.FloatTensor(attrs)
        feats = [torch.FloatTensor(feat) for feat in feats]

        attr_list = [len(view.x[0]) for view in views]

        # 256
        model = MVGAD(feat_size = attrs.size(1), hidden_dim=args.hidden_dim, in_channels_views=attr_list, device=args.device, suppression_factor=args.suppression_factor, enhancement_factor=args.enhancement_factor)

        # adj = adj.to(args.device)
        adj_label = adj_label.to(args.device)
        attrs = attrs.to(args.device)
        edge_index = edge_index.to(args.device)
        model.to(args.device)
        feats = [feat.to(args.device) for feat in feats]
        # books 0.1
        alpha = args.alpha#torch.tensor(args.alpha, requires_grad=True, device=args.device)
        beta =  args.beta#torch.tensor(args.beta, requires_grad=True, device=args.device)
        # alpha = torch.tensor(args.alpha, requires_grad=True, device=args.device)
        # beta =  torch.tensor(args.beta, requires_grad=True, device=args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        epochs = 100
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            A_hat, X_hat, original_embedding, aggregated_embedding = model(attrs, edge_index, feats)
            loss, attr_loss, struct_loss, agg_loss = adaptive_loss_function(A_hat, X_hat, attrs, adj_label, original_embedding, aggregated_embedding, alpha, beta)
        
            l = torch.mean(loss)
            l.backward()

            optimizer.step()
            if epoch == 99:
                model.eval()
                A_hat, X_hat, original_embedding, aggregated_embedding = model(attrs, edge_index, feats)
                loss, attr_loss, struct_loss, agg_loss = adaptive_loss_function(A_hat, X_hat, attrs, adj_label, original_embedding, aggregated_embedding, alpha, beta)
                score = loss.detach().cpu().numpy()

                print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))
            
                res_list.append(roc_auc_score(label, score))
    # 将列表转换为 numpy 数组
    arr = np.array(res_list)

    # 计算平均值
    mean = np.mean(arr)

    # 计算标准差
    std_dev = np.std(arr)

    print("Mean:", mean)
    print("Standard Deviation:", std_dev)
    print(f'{mean*100:.2f}'+"±"+f'{std_dev:.2f}')
    if args.flag:
        # with open(f"{args.dataset}.csv", "a") as f:
        #     f.write(f'{args.hidden_dim},{args.alpha},{args.beta}:{mean*100:.2f}'"±"f'{std_dev:.2f}'+"\n")
        # with open(f"enhancement_factor.csv", "a") as f:
        #     f.write(f'{args.dataset},{args.enhancement_factor}:{mean*100:.2f}'"±"f'{std_dev:.2f}'+"\n")
        # with open(f"suppression_factor.csv", "a") as f:
        #     f.write(f'{args.dataset},{args.suppression_factor}:{mean*100:.2f}'"±"f'{std_dev:.2f}'+"\n")
        with open(f"ablation.csv", "a") as f:
            f.write(f'{args.dataset}:{mean*100:.2f}'"±"f'{std_dev:.2f}'+"\n")
        # with open(f"views.csv", "a") as f:
        #     f.write(f'{args.dataset}:{mean*100:.2f}'"±"f'{std_dev:.2f}'+"\n")
        # with open(f"{args.dataset}.csv", "a") as f:
        #     f.write(f'{args.suppression_factor},{args.enhancement_factor}:{mean*100:.2f}'"±"f'{std_dev:.2f}'+"\n")
        # with open(f"{args.dataset}.csv", "a") as f:
        #     f.write(f'{args.alpha},{args.beta}:{mean*100:.2f}'"±"f'{std_dev:.2f}'+"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default="weibo", help='dataset name: inj_cora/inj_amazon/books/reddit/weibo/disney/enron')
    # parser.add_argument('--hidden_dim', type=int, default=256, help='dimension of hidden embedding (default: 64)')
    # parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    # parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    # parser.add_argument('--alpha', type=float, default=.99, help='balance parameter')
    # parser.add_argument('--beta', type=float, default=0., help='balance parameter')
    # parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    # parser.add_argument('--flag', default='False', type=str, help='w/o write to csv')

    # args = parser.parse_args()
    # print(args)
    # train_mvgad(args)
    # raise
    alpha_list = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    beta_list = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    hidden_dim_list = [32,64,128,256, 512]
    datasets = ['inj_cora', 'books', 'reddit', 'disney', 'enron']#, 'weibo']
    datasets =  ['enron', 'books']
    # datasets = ['disney', 'reddit']
    datasets = ['disney'] #
    # 抑制因子（Suppression Factor） - 这个名称强调了该参数的作用是抑制或减少某些视图的权重，特别是那些被认为含有较多噪声或互斥信息的视图。
    # 增强因子（Enhancement Factor） - 这个名称突出了该参数的功能是对重要性较高的视图进行加强。
    # suppression_factor = 2.
    # enhancement_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # suppression_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # lrs = [2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6]
    # for dataset in datasets:
    #     for suppression_factor in suppression_factors:
    #         parser = argparse.ArgumentParser()
    #         parser.add_argument('--dataset', default=dataset, help='dataset name: inj_cora/inj_amazon/books/reddit/weibo/disney/enron')
    #         parser.add_argument('--hidden_dim', type=int, default=hyperParameter[dataset]['hidden_dim'], help='dimension of hidden embedding (default: 64)')
    #         parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    #         parser.add_argument('--lr', type=float, default=hyperParameter[dataset]['lr'], help='learning rate')
    #         parser.add_argument('--alpha', type=float, default=hyperParameter[dataset]['alpha'], help='balance parameter')
    #         parser.add_argument('--beta', type=float, default=hyperParameter[dataset]['beta'], help='balance parameter')
    #         parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    #         parser.add_argument('--flag', default='True', type=str, help='w/o write to csv')
    #         parser.add_argument('--suppression_factor', type=float, default=suppression_factor, help='w/o write to csv')
    #         parser.add_argument('--enhancement_factor', type=float, default=hyperParameter[dataset]['enhancement_factor'], help='w/o write to csv')

    #         args = parser.parse_args()
    #         print(args)
    #         train_mvgad(args)
    # for dataset in datasets:
    #     for enhancement_factor in enhancement_factors:
    #         parser = argparse.ArgumentParser()
    #         parser.add_argument('--dataset', default=dataset, help='dataset name: inj_cora/inj_amazon/books/reddit/weibo/disney/enron')
    #         parser.add_argument('--hidden_dim', type=int, default=hyperParameter[dataset]['hidden_dim'], help='dimension of hidden embedding (default: 64)')
    #         parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    #         parser.add_argument('--lr', type=float, default=hyperParameter[dataset]['lr'], help='learning rate')
    #         parser.add_argument('--alpha', type=float, default=hyperParameter[dataset]['alpha'], help='balance parameter')
    #         parser.add_argument('--beta', type=float, default=hyperParameter[dataset]['beta'], help='balance parameter')
    #         parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    #         parser.add_argument('--flag', default='True', type=str, help='w/o write to csv')
    #         parser.add_argument('--suppression_factor', type=float, default=hyperParameter[dataset]['suppression_factor'], help='w/o write to csv')
    #         parser.add_argument('--enhancement_factor', type=float, default=enhancement_factor, help='w/o write to csv')

    #         args = parser.parse_args()
    #         print(args)
    #         train_mvgad(args)
    # hidden_him
    # for dataset in datasets:
    #     for hidden_dim in hidden_dim_list:
    #         parser = argparse.ArgumentParser()
    #         parser.add_argument('--dataset', default=dataset, help='dataset name: inj_cora/inj_amazon/books/reddit/weibo/disney/enron')
    #         parser.add_argument('--hidden_dim', type=int, default=hidden_dim, help='dimension of hidden embedding (default: 64)')
    #         parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    #         parser.add_argument('--lr', type=float, default=hyperParameter[dataset]['lr'], help='learning rate')
    #         parser.add_argument('--alpha', type=float, default=hyperParameter[dataset]['alpha'], help='balance parameter')
    #         parser.add_argument('--beta', type=float, default=hyperParameter[dataset]['beta'], help='balance parameter')
    #         parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    #         parser.add_argument('--flag', default='True', type=str, help='w/o write to csv')
    #         parser.add_argument('--suppression_factor', type=float, default=hyperParameter[dataset]['suppression_factor'], help='w/o write to csv')
    #         parser.add_argument('--enhancement_factor', type=float, default=hyperParameter[dataset]['enhancement_factor'], help='w/o write to csv')

    #         args = parser.parse_args()
    #         print(args)
    #         train_mvgad(args)
    # alpha
    # for dataset in datasets:
    #     for alpha in alpha_list:
    #         parser = argparse.ArgumentParser()
    #         parser.add_argument('--dataset', default=dataset, help='dataset name: inj_cora/inj_amazon/books/reddit/weibo/disney/enron')
    #         parser.add_argument('--hidden_dim', type=int, default=hyperParameter[dataset]['hidden_dim'], help='dimension of hidden embedding (default: 64)')
    #         parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    #         parser.add_argument('--lr', type=float, default=hyperParameter[dataset]['lr'], help='learning rate')
    #         parser.add_argument('--alpha', type=float, default=alpha, help='balance parameter')
    #         parser.add_argument('--beta', type=float, default=hyperParameter[dataset]['beta'], help='balance parameter')
    #         parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    #         parser.add_argument('--flag', default='True', type=str, help='w/o write to csv')
    #         parser.add_argument('--suppression_factor', type=float, default=hyperParameter[dataset]['suppression_factor'], help='w/o write to csv')
    #         parser.add_argument('--enhancement_factor', type=float, default=hyperParameter[dataset]['enhancement_factor'], help='w/o write to csv')

    #         args = parser.parse_args()
    #         print(args)
    #         train_mvgad(args)
    # # beta
    # for dataset in datasets:
    #     for beta in beta_list:
    #         parser = argparse.ArgumentParser()
    #         parser.add_argument('--dataset', default=dataset, help='dataset name: inj_cora/inj_amazon/books/reddit/weibo/disney/enron')
    #         parser.add_argument('--hidden_dim', type=int, default=hyperParameter[dataset]['hidden_dim'], help='dimension of hidden embedding (default: 64)')
    #         parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    #         parser.add_argument('--lr', type=float, default=hyperParameter[dataset]['lr'], help='learning rate')
    #         parser.add_argument('--alpha', type=float, default=hyperParameter[dataset]['alpha'], help='balance parameter')
    #         parser.add_argument('--beta', type=float, default=beta, help='balance parameter')
    #         parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    #         parser.add_argument('--flag', default='True', type=str, help='w/o write to csv')
    #         parser.add_argument('--suppression_factor', type=float, default=hyperParameter[dataset]['suppression_factor'], help='w/o write to csv')
    #         parser.add_argument('--enhancement_factor', type=float, default=hyperParameter[dataset]['enhancement_factor'], help='w/o write to csv')
    #         print(args)
    #         args = parser.parse_args()

    #         train_mvgad(args)
    # ablation
    for dataset in datasets:
        # for lr in lrs:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default=dataset, help='dataset name: inj_cora/inj_amazon/books/reddit/weibo/disney/enron')
        parser.add_argument('--hidden_dim', type=int, default=hyperParameter[dataset]['hidden_dim'], help='dimension of hidden embedding (default: 64)')
        parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
        parser.add_argument('--lr', type=float, default=hyperParameter[dataset]['lr'], help='learning rate')

        parser.add_argument('--alpha', type=float, default=hyperParameter[dataset]['alpha'], help='balance parameter')
        parser.add_argument('--beta', type=float, default=hyperParameter[dataset]['beta'], help='balance parameter')
        parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
        parser.add_argument('--flag', default='True', type=str, help='w/o write to csv')
        parser.add_argument('--suppression_factor', type=float, default=hyperParameter[dataset]['suppression_factor'], help='w/o write to csv')
        parser.add_argument('--enhancement_factor', type=float, default=hyperParameter[dataset]['enhancement_factor'], help='w/o write to csv')
        args = parser.parse_args()
        print(args)

        train_mvgad(args)

    # for dataset in datasets:
    #     for alpha in alpha_list:
    #         for beta in beta_list:
    #             parser = argparse.ArgumentParser()
    #             parser.add_argument('--dataset', default=dataset, help='dataset name: inj_cora/inj_amazon/books/reddit/weibo/disney/enron')
    #             parser.add_argument('--hidden_dim', type=int, default=hyperParameter[dataset]['hidden_dim'], help='dimension of hidden embedding (default: 64)')
    #             parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    #             parser.add_argument('--lr', type=float, default=hyperParameter[dataset]['lr'], help='learning rate')

    #             parser.add_argument('--alpha', type=float, default=alpha, help='balance parameter')
    #             parser.add_argument('--beta', type=float, default=beta, help='balance parameter')
    #             parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    #             parser.add_argument('--flag', default='True', type=str, help='w/o write to csv')
    #             parser.add_argument('--suppression_factor', type=float, default=hyperParameter[dataset]['suppression_factor'], help='w/o write to csv')
    #             parser.add_argument('--enhancement_factor', type=float, default=hyperParameter[dataset]['enhancement_factor'], help='w/o write to csv')
    #             args = parser.parse_args()
    #             print(args)

    #             train_mvgad(args)
