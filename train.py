#%%
# import dgl
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


from load_data import *
from fairdla import *
# from fairadg import *
from utils import *
import torch.nn as nn
from torch_sparse import SparseTensor
import logging



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german','income'])
    parser.add_argument("--num_heads", type=int, default=1, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    # 尝试改为2，敏感和非敏感
    parser.add_argument("--channels", type=int, default=2, help="number of channels")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf',
                                                                     'RobustGCN', 'mlpgcn', 'gcnori', 'disengnn',
                                                                     'disengcn', 'pcagcn', 'adagcn', 'adagcn_new'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--tem', type=float, default=0.5, help='the temperature of contrastive learning loss '
                                                               '(mutual information maximize)')
    parser.add_argument('--alpha', type=float, default=0.25, help='weight coefficient for disentanglement')
    parser.add_argument('--beta', type=float, default=0.25, help='weight coefficient for channel masker')
    parser.add_argument('--lr_w', type=float, default=1,
                        help='the learning rate of the adaptive weight coefficient')
    parser.add_argument('--model_type', type=str, default='gnn', choices=['gnn', 'mlp', 'other'])
    parser.add_argument('--weight_path', type=str, default='./Weights/model_weight.pt')
    parser.add_argument('--save_results', type=bool, default=True)
    parser.add_argument('--pre_seed', type=int, default=1)
    parser.add_argument('--drl', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--avgy', type=str2bool, default=False)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--copy', type=int, default=0)
    parser.add_argument('--adv', type=int, default=0)
    args = parser.parse_known_args()[0]
        # 2. 设置设备
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print(args.device)
        torch.cuda.set_device(args.device)
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    # if args.cuda:
    #     print("CUDA is available and will be used.")
    # else:
    #     print("CUDA is not available or is disabled. Using CPU.")

    # # set device
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run(args,logger):
    torch.set_printoptions(threshold=float('inf'))
    """
    Load data
    """
    # 敏感属性和最后的分类都是二元分类，个数都应该填1
    fair_dataset = FairDataset(args.dataset, args.device)
    fair_dataset.load_data()
    # print(fair_dataset.labels.shape)
    # label_counts = np.bincount(fair_dataset.labels.detach().cpu().numpy())
    # label_ratios = label_counts / len(fair_dataset.labels)
    # print(label_ratios)
    
    
    num_class = 1
    args.nfeat = fair_dataset.features.shape[1]
    args.nnode = fair_dataset.features.shape[0]
    args.nclass = num_class
    # print(num_class)

    """
    Build model
    """
    print(args.drl)
    if args.drl ==1:
        fairsad_pretrainer = FairDLA_drl(args,logger)
    
        """
        drl model 
        """
        auc_roc_test, f1_s_test, acc_test, parity_test, equality_test = fairsad_pretrainer.train_fit(fair_dataset, args.epochs, alpha=args.alpha,
                                                                                                beta=args.beta, pbar=args.pbar,logger=logger)
        
        


    if args.drl == 0:
        print('re')
        #RE
        print(args.rs)
        print('avg',args.avgy)

        fairsad_trainer = FairDLA_dra(args,logger)
        weight_path = f"{args.weight_path}_{args.dataset}.pth"
        checkpoint = torch.load(weight_path)
        fairsad_trainer.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        fairsad_trainer.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        fairsad_trainer.channel_cls.load_state_dict(checkpoint['channel_cls_state_dict'])
        embeddings = fairsad_trainer.generate_embeddings(fair_dataset)

     

        auc_roc_test, f1_s_test, acc_test, parity_test, equality_test = fairsad_trainer.train_fit(fair_dataset, args.epochs, alpha=args.alpha,
                                                                                                beta=args.beta, pbar=args.pbar,logger=logger)


    return auc_roc_test, f1_s_test, acc_test, parity_test, equality_test


if __name__ == '__main__':
    # Training settings
    args = args_parser()

        # 创建一个文件处理器，将日志输出到文件
        # 创建日志目录（如果不存在）
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    # 根据 datasetname 创建日志文件路径
    log_file = os.path.join(log_dir, f'{args.dataset}_training_log.txt')

    # 配置日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个文件处理器，将日志输出到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 获取logger实例，并添加文件处理器
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    model_num = 1
    results = Results(args.seed_num, model_num, args)
    if args.drl == 1:
        args.pbar = tqdm(total=args.epochs, desc=f"Seed {args.pre_seed + 1}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        set_seed(args.pre_seed)

        # running train
        results.auc[0, :], results.f1[0, :], results.acc[0, :], results.parity[0, :], results.equality[0, :] = run(args, logger)

    else:
        # retarin
        # print('ree')
        # print(args.seed_num)
        for seed in range(args.seed_num):
            
            print(seed)
            # set seeds
            args.pbar = tqdm(total=args.epochs, desc=f"Seed {seed + 1}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
            set_seed(seed)

            # running train
            results.auc[seed, :], results.f1[seed, :], results.acc[seed, :], results.parity[seed, :], \
            results.equality[seed, :] = run(args,logger)

    # reporting results
    results.report_results()
    if args.save_results:
        results.save_results(args)
