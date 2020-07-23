import json
import time

import click
import dgl
import numpy as np
import pandas as pd
import torch as th
import torch.optim as optim
import tqdm
from dgl import function as fn
from dgl.nn.pytorch import HeteroGraphConv
from dgl.utils import expand_as_pair
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def prepare_mp(g):
    g["click"].in_degree(0)
    g["click"].out_degree(0)
    g["click"].find_edges([0])
    g["click_by"].in_degree(0)
    g["click_by"].out_degree(0)
    g["click_by"].find_edges([0])


def load_subtensor(blocks, batch_labels, device):
    batch_blocks = [i.to(device) for i in blocks]
    batch_labels = batch_labels.to(device)
    return batch_blocks, batch_labels


def block_add_feat(block, graph):
    # 插入属性
    for srctype, etype, dsttype in block.canonical_etypes:
        if block.number_of_edges(etype) > 0:
            for k in graph.nodes[srctype].data:
                if k in [
                    "user_id", "creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry",
                    "degrees"
                ]:
                    block[etype].srcdata[k] = graph.nodes[srctype].data[k][block[etype].srcdata[dgl.NID]]
            for k in graph.nodes[dsttype].data:
                if k in [
                    "user_id", "creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry",
                    "degrees"
                ]:
                    block[etype].dstdata[k] = graph.nodes[dsttype].data[k][block[etype].dstdata[dgl.NID]]


def get_labels(block, graph):
    age = graph.nodes["user"].data["age"][block["click_by"].dstdata[dgl.NID]] - 1
    gender = graph.nodes["user"].data["gender"][block["click_by"].dstdata[dgl.NID]] - 1
    return age * 2 + gender


class NeighborSampler(object):
    def __init__(self, g, fanouts, replace=True):
        self.g = g
        self.fanouts = fanouts
        self.replace = replace

    def sample_blocks(self, seeds):
        seeds = {
            "user": th.LongTensor(np.asarray(seeds))
        }
        blocks = []
        for fanout in self.fanouts:
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=self.replace)
            block = dgl.to_block(frontier, seeds)
            # seed是被传播的节点，所以src是输入节点
            seeds = {
                "user": block["click"].srcdata[dgl.NID],
                "ad": block["click_by"].srcdata[dgl.NID]
            }
            blocks.insert(0, block)
        # 只在第一层插入属性
        block_add_feat(blocks[0], self.g)
        # 获取最后一层dst的label
        batch_labels = get_labels(blocks[-1], self.g)
        return blocks, batch_labels


class DenseEmbedding(th.nn.Module):
    def __init__(self, N, dim, min=None, max=None, log=None, eps=1e-15):
        super(DenseEmbedding, self).__init__()
        self.N = N
        self.dim = dim
        self.eps = eps
        self.min = min
        self.max = max
        self.log = log
        self.register_buffer('k', (2 * th.arange(N, dtype=th.float32) + 1) / N / 2)
        self.weight = th.nn.Parameter(th.Tensor(N, dim))
        self.reset_parameters()

    def reset_parameters(self):
        th.nn.init.normal_(self.weight)

    def forward(self, query):
        if self.log:
            query = th.log(query)
        if not self.min is None:
            query = (query - self.min) / (self.max - self.min)
        query = query.unsqueeze(-1)
        query = 1 / th.abs(query - self.k + self.eps)
        weight = th.softmax(query, dim=-1)
        return weight @ self.weight

    def extra_repr(self):
        return "%s, %s" % (self.N, self.dim)


class SAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats, gnn_mlps, bias=True):
        super(SAGEConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        # din attention
        self.atten_src = nn.utils.weight_norm(nn.Linear(self._in_src_feats, out_feats), name='weight')
        self.atten_dst = nn.utils.weight_norm(nn.Linear(self._in_src_feats, out_feats), name='weight')
        self.atten_sub = nn.utils.weight_norm(nn.Linear(self._in_src_feats, out_feats), name='weight')
        self.atten_mul = nn.utils.weight_norm(nn.Linear(self._in_src_feats, out_feats), name='weight')
        self.atten_out = nn.utils.weight_norm(nn.Linear(out_feats, 1), name='weight')
        self.leaky_relu = nn.LeakyReLU(0.2)
        # other
        self.fc_pool = nn.utils.weight_norm(nn.Linear(self._in_src_feats, self._in_src_feats), name='weight')
        self.fc_pool2 = nn.utils.weight_norm(nn.Linear(self._in_src_feats, self._in_src_feats), name='weight')
        self.fc_self = nn.utils.weight_norm(nn.Linear(self._in_dst_feats, out_feats, bias=bias), name='weight')
        self.fc_neigh = nn.utils.weight_norm(nn.Linear(self._in_src_feats, out_feats, bias=bias), name='weight')
        self.fc_neigh2 = nn.utils.weight_norm(nn.Linear(self._in_src_feats, out_feats, bias=bias), name='weight')
        # mlps
        self.out_mlp = nn.ModuleList()
        for i in range(gnn_mlps):
            self.out_mlp.append(nn.utils.weight_norm(nn.Linear(out_feats, out_feats), name='weight'))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.atten_src.weight, gain=gain)
        nn.init.xavier_uniform_(self.atten_dst.weight, gain=gain)
        nn.init.xavier_uniform_(self.atten_sub.weight, gain=gain)
        nn.init.xavier_uniform_(self.atten_mul.weight, gain=gain)
        nn.init.xavier_uniform_(self.atten_out.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_pool2.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh2.weight, gain=gain)
        for i in self.out_mlp:
            nn.init.xavier_uniform_(i.weight, gain=gain)

    def forward(self, graph, feat):
        graph = graph.local_var()
        if isinstance(feat, tuple):
            feat_src, feat_dst = feat
        else:
            feat_src = feat_dst = feat
        h_self = feat_dst
        # DIN attention: 两个向量、两个向量的差、两个向量的积，分别mlp到n_hidden，再相加，再mlp到1
        ## 计算两个向量的差和积
        graph.srcdata.update({'e_src': feat_src})
        graph.dstdata.update({'e_dst': feat_dst})
        graph.apply_edges(fn.u_sub_v('e_src', 'e_dst', 'e_sub'))
        graph.apply_edges(fn.u_mul_v('e_src', 'e_dst', 'e_mul'))
        ## 分别mlp
        graph.srcdata["e_src"] = self.atten_src(feat_src)
        graph.dstdata["e_dst"] = self.atten_dst(feat_dst)
        graph.edata["e_sub"] = self.atten_sub(graph.edata["e_sub"])
        graph.edata["e_mul"] = self.atten_mul(graph.edata["e_mul"])
        ## “mlp后相加”代替“concat后mlp”
        graph.edata["e"] = graph.edata.pop("e_sub") + graph.edata.pop("e_mul")
        graph.apply_edges(fn.e_add_u('e', 'e_src', 'e'))
        graph.apply_edges(fn.e_add_v('e', 'e_dst', 'e'))
        graph.srcdata.pop("e_src")
        graph.dstdata.pop("e_dst")
        ## 第一层激活函数
        graph.edata["e"] = F.gelu(graph.edata["e"])
        ## 第二层mlp变换到1
        graph.edata["e"] = self.leaky_relu(self.atten_out(graph.edata["e"]))
        # max pool
        graph.srcdata['h'] = F.gelu(self.fc_pool(feat_src))
        graph.apply_edges(fn.e_mul_u('e', 'h', 'h'))
        graph.update_all(fn.copy_e('h', 'm'), fn.max('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']
        # mean pool
        graph.srcdata['h'] = F.gelu(self.fc_pool2(feat_src))
        graph.apply_edges(fn.e_mul_u('e', 'h', 'h'))
        graph.update_all(fn.copy_e('h', 'm'), fn.mean('m', 'neigh'))
        h_neigh2 = graph.dstdata['neigh']
        # concat
        rst = self.fc_self(h_self) + self.fc_neigh(h_neigh) + self.fc_neigh2(h_neigh2)
        # mlps
        if len(self.out_mlp) > 0:
            for layer in self.out_mlp:
                o = layer(F.gelu(rst))
                rst = rst + o
        return rst


class GNNEncoder(nn.Module):
    def __init__(
            self, n_hidden, dense_emb_N, n_layers, gnn_mlps, activation, gcn_dropout, g, freq_dict, data
    ):
        super(GNNEncoder, self).__init__()
        self.fine_embeded_size = list(data["prone_16_dict"].values())[0].weight.shape[1]
        self.prone_embeded_size = list(data["prone_dict"].values())[0].weight.shape[1]
        self.w2v_embeded_size = list(data["w2v_dict"].values())[0].weight.shape[1]
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        # 有一个16维微调的prone，一个128维prone，一个度的embedding
        self.user_inputs_dims = self.fine_embeded_size + self.prone_embeded_size + n_hidden
        # 有6个16维微调的prone，6个128维prone，6个128维w2v，一个度的embedding
        self.ad_inputs_dims = \
            self.fine_embeded_size * 6 + self.prone_embeded_size * 6 + self.w2v_embeded_size * 6 + n_hidden
        # 微调的embedding
        self.embed_dict = nn.ModuleDict()
        for k in freq_dict:
            if k != "time":
                # 基于index频率，有偏的L2正则系数
                mba_score = np.array(freq_dict[k])
                mba_score = mba_score.sum(axis=0) / (mba_score + 0.5)
                self.register_buffer('mba_%s' % k, th.from_numpy(mba_score).unsqueeze(1).float())
                # 预训练prone作为embedding，并微调
                if k in data["prone_16_dict"]:
                    self.embed_dict[k] = data["prone_16_dict"][k]
                else:
                    self.embed_dict[k] = nn.Embedding(len(freq_dict[k]), n_hidden)
        # prone embedding
        self.prone_embed = data["prone_dict"]
        # w2v embedding
        self.w2v_embed = data["w2v_dict"]
        # dense embedding
        self.degrees_emb = DenseEmbedding(
            dense_emb_N, n_hidden,
            min=np.log(min([g.nodes["user"].data["degrees"].min(), g.nodes["ad"].data["degrees"].min()])),
            max=np.log(max([g.nodes["user"].data["degrees"].max(), g.nodes["ad"].data["degrees"].max()])),
            log=True
        )
        # 输入的MLP
        self.user_mlp = nn.utils.weight_norm(nn.Linear(self.user_inputs_dims, n_hidden), name='weight')
        self.ad_mlp = nn.utils.weight_norm(nn.Linear(self.ad_inputs_dims, n_hidden), name='weight')
        # 卷积层
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            self.layers.append(HeteroGraphConv({
                "click": SAGEConv(n_hidden * (i + 1), n_hidden, gnn_mlps),
                "click_by": SAGEConv(n_hidden * (i + 1), n_hidden, gnn_mlps)
            }, aggregate="sum"))
        # others
        self.gcn_dropout = nn.Dropout(gcn_dropout)
        self.activation = activation

    def freeze(self):
        for param in self.embed_dict.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.embed_dict.parameters():
            param.requires_grad = True

    def embed_l2_loss(self, l2_reg):
        loss = 0
        for k in self.embed_dict:
            loss += 0.5 * l2_reg * th.sum(th.pow(self.embed_dict[k].weight, 2) * self.__getattr__('mba_%s' % k))
        return loss

    def get_embeding(self, block):
        res = {}
        for srctype, etype, dsttype in block.canonical_etypes:
            if block.number_of_edges(etype) > 0:
                if srctype == "user":
                    device = block[etype].srcdata["user_id"].device
                    # src
                    user_feats = []
                    user_feats.append(self.prone_embed["user_id"](block[etype].srcdata["user_id"].cpu()).to(device))
                    user_feats.append(self.embed_dict["user_id"](block[etype].srcdata["user_id"]))
                    user_feats.append(self.degrees_emb(block[etype].srcdata["degrees"]))
                    # dst
                    ad_feats = []
                    for f_name in ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id",
                                   "industry"]:
                        ad_feats.append(self.prone_embed[f_name](block[etype].dstdata[f_name].cpu()).to(device))
                    for f_name in ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id",
                                   "industry"]:
                        ad_feats.append(self.w2v_embed[f_name](block[etype].dstdata[f_name].cpu()).to(device))
                    for f_name in ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id",
                                   "industry"]:
                        ad_feats.append(self.embed_dict[f_name](block[etype].dstdata[f_name]))
                    ad_feats.append(self.degrees_emb(block[etype].dstdata["degrees"]))
                    res[srctype] = (th.cat(user_feats, 1), th.cat(ad_feats, 1))
                else:
                    device = block[etype].dstdata["user_id"].device
                    # src
                    ad_feats = []
                    for f_name in ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id",
                                   "industry"]:
                        ad_feats.append(self.prone_embed[f_name](block[etype].srcdata[f_name].cpu()).to(device))
                    for f_name in ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id",
                                   "industry"]:
                        ad_feats.append(self.w2v_embed[f_name](block[etype].srcdata[f_name].cpu()).to(device))
                    for f_name in ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id",
                                   "industry"]:
                        ad_feats.append(self.embed_dict[f_name](block[etype].srcdata[f_name]))
                    ad_feats.append(self.degrees_emb(block[etype].srcdata["degrees"]))
                    # dst
                    user_feats = []
                    user_feats.append(self.prone_embed["user_id"](block[etype].dstdata["user_id"].cpu()).to(device))
                    user_feats.append(self.embed_dict["user_id"](block[etype].dstdata["user_id"]))
                    user_feats.append(self.degrees_emb(block[etype].dstdata["degrees"]))
                    res[srctype] = (th.cat(ad_feats, 1), th.cat(user_feats, 1))
        return res

    def forward(self, blocks):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l == 0:
                feat = self.get_embeding(blocks[0])
                if "user" in feat:
                    feat["user"] = (self.user_mlp(feat["user"][0]), self.ad_mlp(feat["user"][1]))
                if "ad" in feat:
                    feat["ad"] = (self.ad_mlp(feat["ad"][0]), self.user_mlp(feat["ad"][1]))
                for ntype in feat:
                    feat[ntype] = tuple(self.activation(i) for i in feat[ntype])
                    feat[ntype] = tuple(self.gcn_dropout(i) for i in feat[ntype])
                feat_all = {
                    "user": feat["user"][0],
                    "ad": feat["ad"][0]
                }
            else:
                feat = {}
                for srctype, etype, dsttype in block.canonical_etypes:
                    if block.number_of_edges(etype) > 0:
                        feat[srctype] = (feat_all[srctype][:block[etype].number_of_src_nodes()],
                                         feat_all[dsttype][:block[etype].number_of_dst_nodes()])
            h = layer(block, feat)
            for k in h:
                h[k] = self.activation(h[k])
                h[k] = self.gcn_dropout(h[k])
                # dense-net
                feat_all[k] = th.cat([feat_all[k][:h[k].shape[0]], h[k]], 1)
        return feat_all.get("user"), feat_all.get("ad")

    def inference(self, g, batch_size, device, num_workers=5):
        def get_collate_fn(ntype):
            def collate_fn(seeds):
                batch_nodes = {
                    ntype: th.LongTensor(np.asarray(seeds))
                }
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                return seeds, block

            return collate_fn

        loader_dict = {}
        loader_dict["user"] = DataLoader(
            dataset=list(range(g.number_of_nodes("user"))),
            batch_size=batch_size,
            collate_fn=get_collate_fn("user"),
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers)
        loader_dict["ad"] = DataLoader(
            dataset=list(range(g.number_of_nodes("ad"))),
            batch_size=batch_size,
            collate_fn=get_collate_fn("ad"),
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers)
        x = {}
        for ntype in g.ntypes:
            x[ntype] = th.zeros(g.number_of_nodes(ntype), self.n_hidden)
            for start in tqdm.trange(0, g.number_of_nodes(ntype), batch_size):
                end = start + batch_size
                # 测试的时候只需要把非训练集的lable特征mask掉就行
                if ntype == "user":
                    user_feats = []
                    user_feats.append(self.prone_embed["user_id"](g.nodes[ntype].data["user_id"][start:end]).to(device))
                    user_feats.append(self.embed_dict["user_id"](g.nodes[ntype].data["user_id"][start:end].to(device)))
                    user_feats.append(self.degrees_emb(g.nodes[ntype].data["degrees"][start:end].to(device)))
                    user_feats = self.user_mlp(th.cat(user_feats, 1))
                    user_feats = self.activation(user_feats)
                    user_feats = self.gcn_dropout(user_feats)
                    x[ntype][start:end] = user_feats.cpu()
                else:
                    ad_feats = []
                    for f_name in ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id",
                                   "industry"]:
                        ad_feats.append(self.prone_embed[f_name](g.nodes[ntype].data[f_name][start:end]).to(device))
                    for f_name in ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id",
                                   "industry"]:
                        ad_feats.append(self.w2v_embed[f_name](g.nodes[ntype].data[f_name][start:end]).to(device))
                    for f_name in ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id",
                                   "industry"]:
                        ad_feats.append(self.embed_dict[f_name](g.nodes[ntype].data[f_name][start:end].to(device)))
                    ad_feats.append(self.degrees_emb(g.nodes[ntype].data["degrees"][start:end].to(device)))
                    ad_feats = self.ad_mlp(th.cat(ad_feats, 1))
                    ad_feats = self.activation(ad_feats)
                    ad_feats = self.gcn_dropout(ad_feats)
                    x[ntype][start:end] = ad_feats.cpu()
        feat_all = x
        # 处理卷积
        for l, layer in enumerate(self.layers):
            x = {}
            for ntype in g.ntypes:
                x[ntype] = th.zeros(g.number_of_nodes(ntype), self.n_hidden)
                for seeds, block in tqdm.tqdm(loader_dict[ntype]):
                    th.cuda.empty_cache()
                    block = block.to(device)
                    feat = {}
                    for srctype, etype, dsttype in block.canonical_etypes:
                        if block.number_of_edges(etype) > 0:
                            feat[srctype] = (
                                feat_all[srctype][block[etype].srcdata[dgl.NID]].to(device),
                                feat_all[dsttype][block[etype].dstdata[dgl.NID]].to(device))
                    h = layer(block, feat)
                    for k in h:
                        h[k] = self.activation(h[k])
                        h[k] = self.gcn_dropout(h[k])
                        x[k][seeds] = h[k].cpu()
            feat_all["user"] = th.cat([feat_all["user"], x["user"]], 1)
            feat_all["ad"] = th.cat([feat_all["ad"], x["ad"]], 1)
        return feat_all.get("user"), feat_all.get("ad")


class Net(nn.Module):
    def __init__(
            self, n_hidden, dense_emb_N, n_layers, gnn_mlps, activation, gcn_dropout, g, freq_dict, data
    ):
        super(Net, self).__init__()
        self.encoder = GNNEncoder(
            n_hidden, dense_emb_N, n_layers, gnn_mlps, activation, gcn_dropout, g, freq_dict, data)
        # 输出层
        self.out_layers = nn.ModuleList()
        self.out_layers.append(nn.utils.weight_norm(nn.Linear(n_hidden * (n_layers + 1), 20), name='weight'))

    def freeze(self):
        self.encoder.freeze()

    def unfreeze(self):
        self.encoder.unfreeze()

    def embed_l2_loss(self, l2_reg):
        return self.encoder.embed_l2_loss(l2_reg)

    def forward(self, blocks):
        feat, _ = self.encoder(blocks)
        for layer in self.out_layers:
            out = layer(feat)
            feat = F.gelu(out)
        return out

    def inference(self, g, batch_size, device, num_workers=5):
        feat_all, _ = self.encoder.inference(g, batch_size, device, num_workers=num_workers)
        outs = th.zeros(g.number_of_nodes("user"), 20)
        for start in tqdm.trange(0, g.number_of_nodes("user"), batch_size):
            end = start + batch_size
            feat = feat_all[start:end].to(device)
            for layer in self.out_layers:
                out = layer(feat)
                feat = F.gelu(out)
            outs[start:end] = out.cpu()
        return outs


def train(model, dataloader, device, epoch, l2_reg, lr_scheduler, label_smoothing, optimizer, log_every):
    iter_tput = []
    model.train()
    for step, (blocks, batch_labels) in enumerate(dataloader):
        th.cuda.empty_cache()
        tic_step = time.time()
        seeds = blocks[-1]["click_by"].dstdata[dgl.NID]
        # Load the input features as well as output labels
        blocks, batch_labels = load_subtensor(blocks, batch_labels, device)
        N = batch_labels.size(0)
        smoothed_labels = th.full(size=(N, 20), fill_value=(1 - label_smoothing) / (20 - 1)).to(device)
        smoothed_labels.scatter_(dim=1, index=th.unsqueeze(batch_labels, dim=1), value=label_smoothing)
        # Compute loss and prediction
        batch_pred = model(blocks)
        log_prob = F.log_softmax(batch_pred, dim=1)
        loss_total = -th.sum(log_prob * smoothed_labels) / N
        # 对embedding加l2正则
        loss = loss_total + model.embed_l2_loss(l2_reg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        iter_tput.append(len(seeds) / (time.time() - tic_step))
        if step % log_every == 0:
            batch_pred = th.argmax(batch_pred, dim=1)
            batch_pred_age = batch_pred // 2
            batch_pred_gender = batch_pred % 2
            batch_label_age = batch_labels // 2
            batch_label_gender = batch_labels % 2
            acc_age = (batch_pred_age == batch_label_age).float().sum().item() / len(batch_pred)
            acc_gender = (batch_pred_gender == batch_label_gender).float().sum().item() / len(batch_pred)
            acc = acc_age + acc_gender
            gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
            print(
                'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Train Acc_age {:.4f} | Train Acc_gender {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                    epoch + 1,
                    step,
                    loss_total.item(),
                    acc,
                    acc_age,
                    acc_gender,
                    np.mean(iter_tput[3:]),
                    gpu_mem_alloc))


def eval(model, dataloader, device, epoch):
    model.eval()
    with th.no_grad():
        age_sum = 0
        gender_sum = 0
        count = 0
        for step, (blocks, batch_labels) in enumerate(dataloader):
            th.cuda.empty_cache()
            blocks, batch_labels = load_subtensor(blocks, batch_labels, device)
            batch_pred = model(blocks)
            batch_pred = th.argmax(batch_pred, dim=1)
            batch_pred_age = batch_pred // 2
            batch_pred_gender = batch_pred % 2
            batch_label_age = batch_labels // 2
            batch_label_gender = batch_labels % 2
            # 累计acc
            age_sum += (batch_pred_age == batch_label_age).float().sum().item()
            gender_sum += (batch_pred_gender == batch_label_gender).float().sum().item()
            count += len(batch_pred)
        acc_age = age_sum / count
        acc_gender = gender_sum / count
        acc = acc_age + acc_gender
        print('Epoch {}: Eval Acc {:.4f} | Eval Acc_age {:.4f} | Eval Acc_gender {:.4f}'.format(
            epoch, acc, acc_age, acc_gender))


def predict(model, g, batch_size, device, val_mask=None, return_all_pob=False):
    model.eval()
    with th.no_grad():
        pred = model.inference(g, batch_size, device)
    if return_all_pob:
        return pred
    pred = pred[val_mask]
    pred = th.argmax(pred, dim=1)
    pred_age = pred // 2
    pred_gender = pred % 2
    return pred_age + 1, pred_gender + 1


def load_data(kf_index):
    prone_dict = {}
    prone_16_dict = {}
    w2v_dict = {}
    # load data
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    with open("data/freq.json", "r", encoding="utf-8") as f:
        freq_dict = json.load(f)
    u2a_mat = np.load("data/u2a_mat.npz")
    feature = np.load("data/feature.npz")
    print("loading edges....")
    g = dgl.heterograph({
        ('user', 'click', 'ad'): tuple(u2a_mat["u2a_mat"].tolist()),
        ('ad', 'click_by', 'user'): tuple(u2a_mat["u2a_mat"][[1, 0],].tolist())
    })
    print("loading edges done.")
    g["click"].edata["click_times"] = th.from_numpy(u2a_mat["click_times_mat"]).float()
    g["click_by"].edata["click_times"] = th.from_numpy(u2a_mat["click_times_mat"]).float()
    # 节点的度
    g.nodes["user"].data['degrees'] = g.in_degrees(etype="click_by").float()
    g.nodes["ad"].data['degrees'] = g.in_degrees(etype="click").float()
    # nodes
    g.nodes["user"].data['user_id'] = th.arange(g.number_of_nodes("user")).long()
    g.nodes["user"].data["age"] = th.from_numpy(feature["age"]).long()
    g.nodes["user"].data["gender"] = th.from_numpy(feature["gender"]).long()
    g.nodes["ad"].data['creative_id'] = th.arange(g.number_of_nodes("ad")).long()
    g.nodes["ad"].data["ad_id"] = th.from_numpy(feature["ad_id"]).long()
    g.nodes["ad"].data["product_id"] = th.from_numpy(feature["product_id"]).long()
    g.nodes["ad"].data["product_category"] = th.from_numpy(feature["product_category"]).long()
    g.nodes["ad"].data["advertiser_id"] = th.from_numpy(feature["advertiser_id"]).long()
    g.nodes["ad"].data["industry"] = th.from_numpy(feature["industry"]).long()
    prepare_mp(g)
    print(g)
    # 加载ProNE预训练向量和用于微调的16维prone向量
    prone = np.load("data/tencent_128_-1.0.npy")
    prone_16 = np.load("data/tencent_16_-0.6.npy")
    n = 0
    for feat in [
        "user_id", "creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry"
    ]:
        m = len(vocab_dict[feat])
        prone_dict[feat] = nn.Embedding.from_pretrained(th.from_numpy(prone[n: n + m]).float(), freeze=True)
        prone_16_dict[feat] = nn.Embedding.from_pretrained(th.from_numpy(prone_16[n: n + m]).float(), freeze=False)
        n += m
    print("load prone embedding done.")
    # 加载w2v向量
    for feat in [
        # "user_id",
        "creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry"
    ]:
        w2v_dict[feat] = nn.Embedding.from_pretrained(th.from_numpy(np.load("data/w2v_%s.npy" % feat)).float(),
                                                      freeze=True)
    print("load word2vec embedding done.")
    # k折
    train_mask = (th.IntTensor(feature["kf_index"]) != kf_index) & (th.IntTensor(feature["kf_index"]) > 0)
    val_mask = th.IntTensor(feature["kf_index"]) == kf_index
    test_mask = th.IntTensor(feature["kf_index"]) == 0
    train_nid = th.nonzero(train_mask)[:, 0]
    val_nid = th.nonzero(val_mask)[:, 0]
    test_nid = th.nonzero(test_mask)[:, 0]
    g.nodes["user"].data["train_mask"] = train_mask
    data = {
        "prone_dict": prone_dict,
        "prone_16_dict": prone_16_dict,
        "w2v_dict": w2v_dict
    }
    return g, vocab_dict, freq_dict, data, val_mask, test_mask, train_nid, val_nid, test_nid


@click.command()
@click.option("--kf_index", default=1)
@click.option("--num_hidden", default=128)
@click.option("--gpu", default=0)
@click.option("--has_eval", default=True, type=click.BOOL)
def main(kf_index=1, num_hidden=128, gpu=0, has_eval=True):
    print(kf_index, num_hidden, gpu, has_eval)
    th.backends.cudnn.benchmark = True
    th.backends.cudnn.deterministic = False
    th.backends.cudnn.enabled = True
    # 训练参数
    num_epochs = 11
    batch_size = 2048
    test_batch_size = 8192
    lr = 0.045
    # 采样参数
    fan_out = [25, 20]
    num_workers = 10
    sampler_replace = False
    # 模型参数
    dense_emb_N = 100
    num_layers = 2
    gnn_mlps = 1
    gcn_dropout = 0.2
    activation = F.gelu
    embeded_l2_reg = 1.5e-10
    label_smoothing = 0.9
    # 日志参数
    log_every = 20

    if gpu >= 0:
        device = th.device('cuda:%d' % gpu)
    else:
        device = th.device('cpu')
    g, vocab_dict, freq_dict, data, val_mask, test_mask, train_nid, val_nid, test_nid = load_data(kf_index)

    # Define model and optimizer
    model = Net(
        num_hidden, dense_emb_N, num_layers, gnn_mlps, activation, gcn_dropout, g=g, freq_dict=freq_dict, data=data
    )
    model = model.to(device)
    print(model)
    # Create sampler
    sampler = NeighborSampler(g, fan_out, sampler_replace)
    # Create PyTorch DataLoader for constructing blocks
    # collate_fn 参数指定了 sampler，可以对 batch 中的节点进行采样
    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers)
    dataloader_val = DataLoader(
        dataset=val_nid.numpy(),
        batch_size=batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = th.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=num_epochs)
    # train
    avg = 0
    model.freeze()
    for epoch in range(num_epochs):
        if epoch == 1:
            model.unfreeze()
        tic = time.time()
        train(model, dataloader, device, epoch, embeded_l2_reg, lr_scheduler, label_smoothing, optimizer, log_every)
        toc = time.time()
        print('Epoch:{} Time(s): {:.4f}'.format(epoch + 1, toc - tic))
        if has_eval:
            eval(model, dataloader_val, device, epoch + 1)
        if epoch >= 5:
            avg += toc - tic
    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    # 输出验针集概率分布，以及对应label（20分类），计算并打印acc
    pred_prob = predict(model, g, test_batch_size, device, return_all_pob=True)
    pred_prob_val = pred_prob[val_mask]
    pred_prob_test = pred_prob[test_mask]
    pred_val = th.argmax(pred_prob_val, dim=1)
    pred_age_val = pred_val // 2
    pred_gender_val = pred_val % 2
    label_age_val = g.nodes["user"].data["age"][val_mask] - 1
    label_gender_val = g.nodes["user"].data["gender"][val_mask] - 1
    label_val = label_age_val * 2 + label_gender_val
    acc_age_val = (pred_age_val == label_age_val).sum().float() / len(label_age_val)
    acc_gender_val = (pred_gender_val == label_gender_val).sum().float() / len(label_gender_val)
    acc_val = acc_age_val + acc_gender_val
    print('Final: Eval Acc {:.4f} | Eval Acc_age {:.4f} | Eval Acc_gender {:.4f}'.format(
        acc_val.item(), acc_age_val.item(), acc_gender_val.item()))

    if not has_eval:
        # 保存各节点预测得到的embedding表
        model.eval()
        with th.no_grad():
            ufeat, ifeat = model.encoder.inference(g, batch_size, device)
        np.savez(
            "data/graphsage_v70_embed_kf%s_%s.npz" % (kf_index, num_hidden),
            ufeat=ufeat.numpy(),
            ifeat=ifeat.numpy()
        )
        # 保存单模的预测结果，注意label输出要+1
        id2vec = dict([(vocab_dict["user_id"][k], k) for k in vocab_dict["user_id"]])
        if kf_index == 1:
            pred_test = th.argmax(pred_prob_test, dim=1)
            pred_age_test = pred_test // 2
            pred_gender_test = pred_test % 2
            res_df = pd.DataFrame({
                "user_id": [id2vec[i] for i in test_nid.numpy().tolist()],
                "predicted_age": (pred_age_test + 1).numpy().tolist(),
                "predicted_gender": (pred_gender_test + 1).numpy().tolist(),
            })
            res_df.to_csv("submission_kf%s.csv" % kf_index, index=False)
        # 保存验针集到csv文件
        val_df = pd.DataFrame({
            "user_id": [id2vec[i] for i in val_nid.numpy().tolist()],
            "label": label_val.tolist()
        })
        for i in range(20):
            val_df["prob_%s" % i] = pred_prob_val[:, i].tolist()
        val_df.to_csv("val_kf_%s.csv" % kf_index, index=False)
        # 保存测试集
        test_df = pd.DataFrame({
            "user_id": [id2vec[i] for i in test_nid.numpy().tolist()],
        })
        for i in range(20):
            test_df["prob_%s" % i] = pred_prob_test[:, i].tolist()
        test_df.to_csv("test_kf_%s.csv" % kf_index, index=False)


if __name__ == '__main__':
    main()
