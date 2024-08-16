import torch.nn as nn
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import *
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
import torch
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})
class FairDLA_drl(torch.nn.Module):
    def __init__(self, train_args,logger):
        super(FairDLA_drl, self).__init__()
        self.args = train_args
        # model of FairADG: encoder,classifier
        
        self.encoder = DisGCN(nfeat=self.args.nfeat,
                                nhid=self.args.hidden,
                                nclass=self.args.nclass,
                                chan_num=self.args.channels,
                                layer_num=2,
                                dropout=self.args.dropout
                                ).to(self.args.device)
        
        self.classifier = nn.Linear(train_args.hidden, train_args.nclass).to(self.args.device)

        self.per_channel_dim = train_args.hidden // train_args.channels
        self.channel_cls = nn.Linear(self.per_channel_dim, train_args.channels).to(self.args.device)

        # optumizer
        self.optimizer_g = torch.optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()) , lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)

        self.optimizer_c = torch.optim.Adam(list(self.channel_cls.parameters()), lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)

        # loss function
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_dc = DistCor()
        self.criterion_mul_cls = nn.CrossEntropyLoss()
      

        self.encoder.init_parameters()
        self.encoder.init_edge_weight()
        self.logger = logger
        
        

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        output = self.classifier(h)
        return h, output
    
    def save_model(self, path):
        torch.save({
    'encoder_state_dict': self.encoder.state_dict(),
    'classifier_state_dict': self.classifier.state_dict(),
    'channel_cls_state_dict': self.channel_cls.state_dict()
}, f"{self.args.weight_path}_{self.args.dataset}.pth")
        self.logger.info(f"Model saved to {path}")

    def train_fit(self, data, epochs, **kwargs):
        # parsing parameters
        alpha = kwargs.get('alpha', None)
        beta = kwargs.get('beta', None)
        pbar = kwargs.get('pbar', None)

        best_res_val = 0.0
        save_model = 0 
        # training encoder, assigner, classifier
        for epoch in range(epochs):
            self.encoder.train()
            self.classifier.train()
            # cls是否可用
            self.channel_cls.train()

            self.optimizer_g.zero_grad()
            self.optimizer_c.zero_grad()

            h = self.encoder(data.features, data.edge_index)
            output = self.classifier(h)
            # print(output[1])
            # output = self.encoder.predict(h)

            # downstream tasks loss
            loss_cls_train = self.criterion_bce(output[data.idx_train],
                                                data.labels[data.idx_train].unsqueeze(1).float())

            # channel identification loss
            loss_chan_train = 0
            for i in range(self.args.channels):
                chan_output = self.channel_cls(h[:, i*self.per_channel_dim:(i+1)*self.per_channel_dim])
                chan_tar = torch.ones(chan_output.shape[0], dtype=int)*i
                chan_tar = chan_tar.to(self.args.device)
                loss_chan_train += self.criterion_mul_cls(chan_output, chan_tar)

            # distance correlation loss
            loss_disen_train = 0
            len_per_channel = int(h.shape[1] / self.args.channels)
            for i in range(self.args.channels):
                for j in range(i + 1, self.args.channels):
                    loss_disen_train += self.criterion_dc(
                        h[data.idx_train, i * len_per_channel:(i + 1) * len_per_channel],
                        h[data.idx_train, j * len_per_channel:(j + 1) * len_per_channel])

            

            loss_train = loss_cls_train + alpha * (loss_chan_train + loss_disen_train) 

            loss_train.backward()
            self.optimizer_g.step()
            self.optimizer_c.step()

            # evaluating encoder, assigner, classifier
            self.encoder.eval()
            self.classifier.eval()

            if epoch % 10 == 0:
                with torch.no_grad():
                    h = self.encoder(data.features, data.edge_index)
                    
                    y_output_val = self.classifier(h)
                    y_output_val = y_output_val.detach()
                    y_pred_val = (y_output_val.squeeze() > 0).type_as(data.sens)
                    acc_val = accuracy_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                    roc_val = roc_auc_score(data.labels[data.idx_val].cpu(), y_output_val[data.idx_val].cpu())
                    f1_val = f1_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                    parity, equality = fair_metric(y_pred_val[data.idx_val].cpu().numpy(),
                                                data.labels[data.idx_val].cpu().numpy(),
                                                data.sens[data.idx_val].cpu().numpy())
                    
                    
                    res_val = acc_val + roc_val - parity - equality
                    self.logger.info(f'Epoch {epoch}: Val Accuracy: {acc_val:.4f}, Val ROC AUC: {roc_val:.4f}, Val F1: {f1_val:.4f}, Val Parity: {parity:.4f}, Val Equality: {equality:.4f}')

                    if res_val > best_res_val:
                        best_res_val = res_val
                        """
                            evaluation
                        """
                        acc_test = accuracy_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                        roc_test = roc_auc_score(data.labels[data.idx_test].cpu(), y_output_val[data.idx_test].cpu())
                        f1_test = f1_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                        parity_test, equality_test = fair_metric(y_pred_val[data.idx_test].cpu().numpy(),
                                                    data.labels[data.idx_test].cpu().numpy(),
                                                    data.sens[data.idx_test].cpu().numpy())
                        save_model = epoch
                        model_save_path = f"{self.args.weight_path}_{self.args.dataset}.pth"
                        self.save_model(model_save_path)
                    
            if pbar is not None:
                pbar.set_postfix({'train_total_loss': "{:.2f}".format(loss_train.item()),
                                'cls_loss':  "{:.2f}".format(loss_cls_train.item()),
                                'disen_loss': "{:.2f}".format(loss_disen_train.item()),
                                'val_loss': "{:.2f}".format(res_val), 'save model': save_model})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.logger.info(f'ROC AUC: {roc_test:.4f}, Test F1: {f1_test:.4f}, Test Parity: {parity_test:.4f}, Test Equality: {equality_test:.4f}')

        return roc_test, f1_test, acc_test, parity_test, equality_test

class FairDLA_dra(torch.nn.Module):
    def __init__(self, train_args,logger):
        super(FairDLA_dra, self).__init__()
        self.args = train_args
        # model of FairADG: encoder,classifier
        
        self.encoder = DisGCN(nfeat=self.args.nfeat,
                                nhid=self.args.hidden,
                                nclass=self.args.nclass,
                                chan_num=self.args.channels,
                                layer_num=2,
                                dropout=self.args.dropout
                                ).to(self.args.device)
        
        self.classifier = nn.Linear(train_args.hidden, train_args.nclass).to(self.args.device)

        self.per_channel_dim = train_args.hidden // train_args.channels
        self.channel_cls = nn.Linear(self.per_channel_dim, train_args.channels).to(self.args.device)

        # optumizer
        self.optimizer_g = torch.optim.Adam(list(self.encoder.parameters())+list(self.classifier.parameters()) , lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)
        self.optimizer_gc = torch.optim.Adam(list(self.classifier.parameters()) , lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)

        self.optimizer_c = torch.optim.Adam(list(self.channel_cls.parameters()), lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)

        # loss function
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_dc = DistCor()
        self.criterion_mul_cls = nn.CrossEntropyLoss()
        

        self.encoder.init_parameters()
        self.encoder.init_edge_weight()
        self.logger = logger
        self.alpha_a = self.args.alpah_a
        self.k = self.args.k
        self.ad_cre = self.args.ad_cre
        self.avgy = self.args.avgy
       
        
        

    #     for m in self.modules():
    #         self.weights_init(m)

    # def weights_init(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
       
        output = self.classifier(h)
        # 这里不需要处理一下吗？不需要加入sigmod函数吗，变成0-1之间的吗？
        return h, output
    
    def calc_loss(self, embed: torch.Tensor, embed_adv: torch.Tensor) -> torch.Tensor:
        z_embed_adv = self.classifier(embed_adv)
        z_embed = self.classifier(embed)
        l_2 = torch.linalg.norm(z_embed - z_embed_adv, ord=2, dim=1)
        return l_2
    
    @torch.no_grad()
    def Envs_gen_y(self, embed: torch.Tensor, y: torch.Tensor, sens_attr_vectors: dict):
        assert y.dim() == 1 and sens_attr_vectors is not None

        y_repeated = y.repeat_interleave(self.random_attack_num_samples)
        assert embed.dim() == 2 and embed.size(0) == y.size(0)

        noisy_latents = embed.repeat_interleave(self.random_attack_num_samples, dim=0).clone().detach()
        coeffs = (2 * torch.rand(noisy_latents.shape[0], 1, device=noisy_latents.device) - 1) * self.perturb_epsilon

        sens_attr_vector_tensor = torch.zeros_like(noisy_latents)
        for label in sens_attr_vectors.keys():
            mask = (y_repeated == label)
            sens_attr_vector_tensor[mask] = sens_attr_vectors[label]

        noisy_latents += sens_attr_vector_tensor * coeffs

        return noisy_latents, y_repeated
    
    @torch.no_grad()
    def get_cre_ad_y(self, embed: torch.Tensor, labels: torch.Tensor, attr_vectors_diff: dict) -> torch.Tensor:
        noisy_emb_all = []
        losses_all = []
        for _ in range(self.random_attack_num_samples):
            noisy_emb = embed.clone()
            
            # 为每个样本选择对应标签的敏感属性向量差异
            sens_attr_vector_repeated = torch.stack([attr_vectors_diff[label.item()] for label in labels], dim=0)
            
            coeffs = (2 * torch.rand(embed.shape[0], 1, device=embed.device) - 1) * self.perturb_epsilon
            noisy_emb += sens_attr_vector_repeated * coeffs
            noisy_emb_all.append(noisy_emb)
            loss = self.calc_loss(embed, noisy_emb)
            losses_all.append(loss.clone().detach())
            
        losses_all = torch.stack(losses_all, dim=1)
        _, idx = torch.max(losses_all, dim=1)
        adv_examples = []
        for i, sample_idx in enumerate(idx.cpu().tolist()):
            adv_examples.append(noisy_emb_all[sample_idx][i])
        return torch.stack(adv_examples, 0)


    def train_fit(self, data, epochs, **kwargs):
        # parsing parameters
        # alpha = kwargs.get('alpha', None)
        # control loss_inv and loss_chan
        beta = kwargs.get('beta', None)
        pbar = kwargs.get('pbar', None)


        best_res_val = 0.0
        save_model = 0
        self.sens_train, self.sens_val, self.sens_test = data.sens[data.idx_train], data.sens[data.idx_val], \
            data.sens[data.idx_test]
        pre_emb = self.encoder(data.features, data.edge_index)[data.idx_train]
        if self.avgy:
            self.sens_avg = compute_res_vectors_y(pre_emb,self.sens_train,data.labels[data.idx_train])

        # training encoder, assigner, classifier
        for epoch in range(epochs):
            self.encoder.train()
          
            self.classifier.train()
           
            self.channel_cls.train()

            self.optimizer_g.zero_grad()
            self.optimizer_c.zero_grad()

            h = self.encoder(data.features, data.edge_index)
            if self.ad_cre == 0:

                noisy_embeds, y_repeated = self.Envs_gen_y(h[data.idx_train], data.labels[data.idx_train],self.sens_avg)


           
                train_emb = torch.cat([h[data.idx_train],noisy_embeds])
                
                # 对带标签的数据进行了增广，从18876-19867
                # print(train_emb.shape)
                y_targets = torch.cat([data.labels[data.idx_train], y_repeated])



                output = self.classifier(train_emb)

                loss_inv = self.criterion_bce(output,
                                                    y_targets.unsqueeze(1).float())
            else:
                # 先构造对抗样本集，在样本集中选择损失最大即距离最远的作为对抗样本实现
                if self.avgy:
                    train_emb_adv = self.get_adv_examples_y(h[data.idx_train],data.labels[data.idx_train],self.sens_avg)
                else:
                    train_emb_adv = self.get_adv_examples(h[data.idx_train],self.sens_avg)
                # 后面尝试是否要使用对抗样本再相同y的交叉熵损失
                # output = self.classifier(train_emb_adv)
                # adv_loss = self.calc_loss(embeddings, train_embeddings_adv).mean()
                
                z_embed_adv = self.classifier(train_emb_adv)
                # 需要将h中的训练集数据选出来
                z_embed = self.classifier(h[data.idx_train])
                loss_ood = torch.linalg.norm(z_embed - z_embed_adv, ord=2, dim=1).mean()
                loss_inv = self.criterion_bce(z_embed, data.labels[data.idx_train].unsqueeze(1).float()) + 0.7*loss_ood
                # print('cls',loss_cls_train)
        




            # loss_cls_train = self.criterion_bce(output[data.idx_train],
            #                                     data.labels[data.idx_train].unsqueeze(1).float())
            # loss_cls_train = F.binary_cross_entropy_with_logits(output, y_targets.unsqueeze(1).float())

            # channel identification loss
            loss_chan_train = 0
            for i in range(self.args.channels):
                chan_output = self.channel_cls(h[:, i*self.per_channel_dim:(i+1)*self.per_channel_dim])
                chan_tar = torch.ones(chan_output.shape[0], dtype=int)*i
                chan_tar = chan_tar.to(self.args.device)
                loss_chan_train += self.criterion_mul_cls(chan_output, chan_tar)

            # distance correlation loss
            loss_disen_train = 0
            len_per_channel = int(h.shape[1] / self.args.channels)
            for i in range(self.args.channels):
                for j in range(i + 1, self.args.channels):
                    loss_disen_train += self.criterion_dc(
                        h[data.idx_train, i * len_per_channel:(i + 1) * len_per_channel],
                        h[data.idx_train, j * len_per_channel:(j + 1) * len_per_channel])

            # masker loss
            
            #loss_fair = self.compute_fairness_loss(h[data.idx_train], data.sens[data.idx_train])
            # print(loss_fair)
            
            if self.adv == 0:
                loss_train = loss_inv + beta * (loss_chan_train + loss_disen_train) 
            else:
                loss_train = self.ad_cre * loss_inv + beta * (loss_chan_train + loss_disen_train) 

            loss_train.backward()
            self.optimizer_g.step()
            self.optimizer_c.step()

            # evaluating encoder, assigner, classifier
            self.encoder.eval()
            self.classifier.eval()

            if epoch % 10 == 0:
                
                h = self.encoder(data.features, data.edge_index)
                
                y_output_val = self.classifier(h)
                y_output_val = y_output_val.detach()
                y_pred_val = (y_output_val.squeeze() > 0).type_as(data.sens)
                acc_val = accuracy_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                roc_val = roc_auc_score(data.labels[data.idx_val].cpu(), y_output_val[data.idx_val].cpu())
                f1_val = f1_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                parity, equality = fair_metric(y_pred_val[data.idx_val].cpu().numpy(),
                                               data.labels[data.idx_val].cpu().numpy(),
                                               data.sens[data.idx_val].cpu().numpy())

                res_val = acc_val + roc_val - parity - equality
              

                if res_val > best_res_val:
                    best_res_val = res_val
                    """
                        evaluation
                    """
                    acc_test = accuracy_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                    roc_test = roc_auc_score(data.labels[data.idx_test].cpu(), y_output_val[data.idx_test].cpu())
                    f1_test = f1_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                    parity_test, equality_test = fair_metric(y_pred_val[data.idx_test].cpu().numpy(),
                                                   data.labels[data.idx_test].cpu().numpy(),
                                                   data.sens[data.idx_test].cpu().numpy())
                    save_model = epoch
                
            if pbar is not None:
                pbar.set_postfix({'train_total_loss': "{:.2f}".format(loss_train.item()),
                                  'cls_loss':  "{:.2f}".format(loss_inv.item()),
                                  'disen_loss': "{:.2f}".format(loss_disen_train.item()),
                                  
                                  'val_loss': "{:.2f}".format(res_val), 'save model': save_model})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.logger.info(f'ROC AUC: {roc_test:.4f}, Test F1: {f1_test:.4f}, Test Parity: {parity_test:.4f}, Test Equality: {equality_test:.4f}')
    



        return roc_test, f1_test, acc_test, parity_test, equality_test
        
class DisenLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, channels, reduce=True):
        super(DisenLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        # 这里每个通道的特征是将总维度除通道数
        self.per_channel_dim = self.out_dim // self.channels
        self.reduce = reduce

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(channels):
            if reduce:
                self.lin_layers.append(nn.Linear(in_features=in_dim, out_features=self.per_channel_dim))
                self.conv_layers.append(Linear(in_channels=self.per_channel_dim, out_channels=self.per_channel_dim, bias=False,
                                               weight_initializer='glorot'))
            else:
                self.conv_layers.append(Linear(in_channels=self.in_dim, out_channels=self.per_channel_dim, bias=False,
                                               weight_initializer='glorot'))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.per_channel_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))

    def get_reddim_k(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feat = self.lin_layers[k](x)
            z_feats.append(z_feat)
        return z_feats

    def get_k_feature(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feats.append(x)
        return z_feats

    def forward(self, x, edge_index, edge_weight):
        assert self.channels == edge_weight.shape[1], "axis dimension in direction 1 need to be equal to channels number"
        if self.reduce:
            z_feats = self.get_reddim_k(x)
        else:
            z_feats = self.get_k_feature(x)
        c_feats = []
        for k, layer in enumerate(self.conv_layers):
            # print(k)
            c_temp = layer(z_feats[k])
            edge_index_copy = edge_index.clone()
            if not edge_index_copy.has_value():
                edge_index_copy = edge_index_copy.fill_value(1., dtype=None)
            edge_index_copy.storage.set_value_(edge_index_copy.storage.value() * edge_weight[:, k])
            out = self.propagate(edge_index_copy, x=c_temp)
            if self.bias_list is not None:
                out = out + self.bias_list[k]
            c_feats.append(F.normalize(out, p=2, dim=1))
            # print(c_feats[k].shape)
        
        output = torch.cat(c_feats, dim=1)
        # 获取两个通道的分别表示
        channel_1_representation = c_feats[0]  # 第一个通道的表示
        channel_2_representation = c_feats[1]  # 第二个通道的表示

        # return output,channel_1_representation, channel_2_representation
        return output

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class DisGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, chan_num, layer_num, dropout=0.5):
        super(DisGCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout_rate = dropout
        self.chan_num = chan_num
        self.layer_num = layer_num
        self.edge_weight = None

        # 先初始化边权分配器

        self.assigner = NeiborAssigner(nfeat, chan_num)
        self.disenlayers = nn.ModuleList()
        for i in range(layer_num-1):
            if i == 0:
                self.disenlayers.append(DisenLayer(nfeat, nhid, chan_num))
            else:
                self.disenlayers.append(DisenLayer(nhid, nhid, chan_num))
        self.dropout = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.normal_(item, mean=0, std=1)
    
    def init_edge_weight(self):
        for m in self.assigner.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        assert isinstance(edge_index, SparseTensor), "Expected input is sparse tensor"
        feats_pair = torch.cat([x[edge_index.storage._col, :], x[edge_index.storage._row, :]], dim=1)
        edge_weight = self.assigner(feats_pair.detach())
        # print('ew',edge_weight.shape)
        for layer in self.disenlayers:
            # x,hs,hy = layer(x, edge_index, edge_weight)
            x = layer(x, edge_index, edge_weight)
            x = self.dropout(x)
            # hs = self.dropout(hs)
            # hy = self.dropout(hy)
        return x
    

class NeiborAssigner(nn.Module):
    def __init__(self, nfeats, channels):
        super(NeiborAssigner, self).__init__()
       
        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * nfeats, out_features=channels),
            nn.Linear(in_features=channels, out_features=channels)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features_pair):
        # print(features_pair.shape)
        alpha_score = self.layers(features_pair)
        alpha_score = torch.softmax(alpha_score, dim=1)
        # print(alpha_score)
        return alpha_score
