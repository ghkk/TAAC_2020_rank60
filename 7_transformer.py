import json
import os
import random
import time

import click
import numpy as np
import pandas as pd
import torch as th
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class PandasSample(object):
    def __init__(self, pre_user_embed, pre_ad_embed, ad_feats, w2v_dict, kf_index_list, max_len):
        self.df = []
        for kf in kf_index_list:
            self.df.append(pd.read_pickle("data/data_kf_%s.pkl" % kf))
        self.df = pd.concat(self.df)
        self.max_len = max_len
        self.pre_embed = {
            "pre_user_embed": pre_user_embed,
            "pre_ad_embed": pre_ad_embed
        }
        self.ad_feats = ad_feats
        self.w2v_dict = w2v_dict

    def get_seeds(self):
        return np.arange(self.df.shape[0])

    def sampler_len(self, x):
        if len(x["creative_id"]) > self.max_len:
            ids = random.sample([i for i in range(len(x["creative_id"]))], self.max_len)
            return pd.Series({
                "creative_id": [x["creative_id"][i] for i in ids],
                "click_times": [x["click_times"][i] for i in ids]
            })
        else:
            return pd.Series({
                "creative_id": x["creative_id"] + [0 for i in range(self.max_len - len(x["creative_id"]))],
                "click_times": x["click_times"] + [0 for i in range(self.max_len - len(x["creative_id"]))]
            })

    def sampler(self, seeds):
        df = self.df.iloc[seeds,]
        # user_id
        user_id = th.from_numpy(df["user_id"].to_numpy()).long()
        # 判断是否超过最大长度
        df_t = df.apply(self.sampler_len, axis=1)
        # creative_id_list
        creative_id_list = th.from_numpy(np.array(df_t["creative_id"].to_list())).long()
        # click_times
        click_times = th.from_numpy(np.array(df_t["click_times"].to_list())).float()
        # embed
        user_embed = []
        user_embed.append(self.pre_embed["pre_user_embed"](user_id))
        user_embed = th.cat(user_embed, -1)
        ad_embed = []
        ad_embed.append(self.pre_embed["pre_ad_embed"](creative_id_list))
        ad_embed.append(self.w2v_dict["creative_id"](creative_id_list))
        for f_name in self.ad_feats:
            feat = self.ad_feats[f_name][creative_id_list]
            ad_embed.append(self.w2v_dict[f_name](feat))
        ad_embed = th.cat(ad_embed, -1)
        # list_len
        list_len = th.from_numpy(
            df["creative_id"].apply(lambda x: self.max_len if len(x) > self.max_len else len(x)).to_numpy()).long()
        # label
        age = th.from_numpy(df["age"].to_numpy()).long() - 1
        gender = th.from_numpy(df["gender"].to_numpy()).long() - 1
        label = age * 2 + gender
        return (user_id, user_embed, ad_embed, click_times, list_len), label


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


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        weight = th.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(th.zeros(step_dim))

    def forward(self, x, mask=None):
        eij = th.mm(
            x.contiguous().view(-1, self.feature_dim),
            self.weight
        ).view(-1, self.step_dim)
        if self.bias:
            eij = eij + self.b
        eij = th.tanh(eij)
        a = th.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / th.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * th.unsqueeze(a, -1)
        return th.sum(weighted_input, 1)


class Caps_Layer(nn.Module):
    def __init__(self, input_dim_capsule, dim_capsule, num_capsule=1, routings=3, **kwargs):
        super(Caps_Layer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.W = nn.Parameter(
            nn.init.xavier_normal_(th.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))

    def forward(self, x):
        u_hat_vecs = th.matmul(x, self.W)
        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # 转成(batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = th.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)
        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.squash(th.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = th.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
        return outputs.view(-1, self.num_capsule * self.dim_capsule)  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = th.sqrt(s_squared_norm + 1e-7)
        return x / scale


class Net(nn.Module):
    def __init__(
            self, dense_emb_N, dropout=0.0, max_len=200, user_embed_size=384, ad_embed_size=256, num_hidden=512
    ):
        super(Net, self).__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask_base", th.arange(0, max_len + 1).unsqueeze(0))
        self.click_times_emb = DenseEmbedding(dense_emb_N, 16, min=0.0, max=7.0, log=True, eps=1e-15)
        self.user_mlp = nn.Linear(user_embed_size, num_hidden)
        self.ad_mlp = nn.Linear(ad_embed_size + 128 * 6 + 16, num_hidden)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=num_hidden, nhead=8, dim_feedforward=num_hidden * 2, dropout=dropout, activation="gelu"),
            num_layers=1
        )
        self.atten = Attention(num_hidden, max_len + 1)
        self.caps = Caps_Layer(num_hidden, num_hidden)
        self.out_layers = nn.ModuleList()
        self.out_layers.append(
            nn.utils.weight_norm(nn.Linear(num_hidden * 5, 20), name='weight'))

    def forward(self, inputs):
        user_id, user_embed, ad_embed, click_times, list_len = inputs
        user_embed = self.user_mlp(user_embed)
        ad_embed = th.cat([ad_embed, self.click_times_emb(click_times)], -1)
        ad_embed = self.ad_mlp(ad_embed)
        # user和ad拼到一起
        embed = th.cat([user_embed.unsqueeze(1), ad_embed], 1)
        list_len = list_len + 1
        # transformer
        mask = self.mask_base.expand(list_len.shape[0], self.max_len + 1).lt(list_len.unsqueeze(1))
        out = self.encoder(embed.permute(1, 0, 2), src_key_padding_mask=~mask).permute(1, 0, 2)
        # first
        out_first = out[:, 0, :]
        # max
        out_max = th.max(out, dim=1)[0]
        # mean
        out_mean = th.sum(out, dim=1) / list_len.unsqueeze(1)
        # attention
        out_atten = self.atten(out)
        # capsule
        out_caps = self.caps(out)
        feat = th.cat([out_first, out_max, out_mean, out_atten, out_caps], dim=-1)
        feat = self.dropout(feat)
        for layer in self.out_layers:
            out = layer(feat)
            feat = F.gelu(out)
        return out


def train(model, dataloader, device, epoch, lr_scheduler, label_smoothing, optimizer, log_every):
    iter_tput = []
    model.train()
    for step, (batch_input, batch_labels) in enumerate(dataloader):
        th.cuda.empty_cache()
        tic_step = time.time()
        batch_input = [i.to(device) for i in batch_input]
        batch_labels = batch_labels.to(device)
        N = batch_labels.size(0)
        smoothed_labels = th.full(size=(N, 20), fill_value=(1 - label_smoothing) / (20 - 1)).to(device)
        smoothed_labels.scatter_(dim=1, index=th.unsqueeze(batch_labels, dim=1), value=label_smoothing)
        # Compute loss and prediction
        batch_pred = model(batch_input)
        log_prob = F.log_softmax(batch_pred, dim=1)
        loss_total = -th.sum(log_prob * smoothed_labels) / N
        # # 对embedding加l2正则
        loss = loss_total
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        iter_tput.append(batch_labels.shape[0] / (time.time() - tic_step))
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
        for step, (batch_input, batch_labels) in enumerate(dataloader):
            th.cuda.empty_cache()
            batch_input = [i.to(device) for i in batch_input]
            batch_labels = batch_labels.to(device)
            # Compute loss and prediction
            batch_pred = model(batch_input)
            batch_pred = th.argmax(batch_pred, dim=1)
            batch_pred_age = batch_pred // 2
            batch_pred_gender = batch_pred % 2
            batch_label_age = batch_labels // 2
            batch_label_gender = batch_labels % 2
            # 累计acc
            age_sum += (batch_pred_age == batch_label_age).float().sum().cpu().item()
            gender_sum += (batch_pred_gender == batch_label_gender).float().sum().cpu().item()
            count += len(batch_pred)
        acc_age = age_sum / count
        acc_gender = gender_sum / count
        acc = acc_age + acc_gender
        print('Epoch {}: Eval Acc {:.4f} | Eval Acc_age {:.4f} | Eval Acc_gender {:.4f}'.format(
            epoch, acc, acc_age, acc_gender))


def predict(model, dataloader, device):
    model.eval()
    with th.no_grad():
        pred_prob = None
        user_ids = None
        label = None
        for step, (batch_input, batch_labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            th.cuda.empty_cache()
            user_id = batch_input[0]
            batch_input = [i.to(device) for i in batch_input]
            # Compute loss and prediction
            batch_pred = model(batch_input).cpu()
            if pred_prob is not None:
                pred_prob = th.cat([pred_prob, batch_pred], dim=0)
                user_ids = th.cat([user_ids, user_id], dim=0)
                label = th.cat([label, batch_labels], dim=0)
            else:
                pred_prob = batch_pred
                user_ids = user_id
                label = batch_labels
    return user_ids, pred_prob, label


@click.command()
@click.option("--kf_index", default=1)
@click.option("--gpu", default=0)
@click.option("--has_eval", default=True, type=click.BOOL)
def main(kf_index=1, gpu=0, has_eval=True):
    print(kf_index, gpu, has_eval)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % gpu
    th.backends.cudnn.benchmark = True
    th.backends.cudnn.deterministic = False
    th.backends.cudnn.enabled = True
    if gpu >= 0:
        device = th.device('cuda:%d' % 0)
    else:
        device = th.device('cpu')
    # 训练参数
    num_epochs = 10
    batch_size = 2048
    lr = 0.0005
    max_len = 50
    num_workers = 10
    # 模型参数
    pre_num_hidden = 256
    num_hidden = 512
    dense_emb_N = 100
    # num_layers = 2
    # gnn_mlps = 1
    # embeded_l2_reg = 1.5e-10
    # att_dnn_dropout = 0.0
    dropout = 0.2
    label_smoothing = 0.9
    # 日志参数
    log_every = 20
    # data
    ## vocab
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    ## 预训练graphsage向量
    graph_ouputs = np.load("data/graphsage_v70_embed_kf%s_%s.npz" % (kf_index, pre_num_hidden))
    pre_user_embed = nn.Embedding.from_pretrained(th.from_numpy(graph_ouputs["ufeat"]))
    pre_ad_embed = nn.Embedding.from_pretrained(th.from_numpy(graph_ouputs["ifeat"][:, :(2 * pre_num_hidden)]))
    ## ad特征
    feature = np.load("data/feature.npz")
    ad_feats = {
        "ad_id": th.from_numpy(feature["ad_id"]).long(),
        "product_id": th.from_numpy(feature["product_id"]).long(),
        "product_category": th.from_numpy(feature["product_category"]).long(),
        "advertiser_id": th.from_numpy(feature["advertiser_id"]).long(),
        "industry": th.from_numpy(feature["industry"]).long()
    }
    ## w2v向量
    w2v_dict = {}
    for feat in [
        "creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry"
    ]:
        w2v_dict[feat] = nn.Embedding.from_pretrained(th.from_numpy(np.load("data/w2v_%s.npy" % feat)).float())

    sampler = PandasSample(pre_user_embed, pre_ad_embed, ad_feats, w2v_dict,
                           [kf for kf in range(1, 6) if kf != kf_index], max_len)
    sampler_val = PandasSample(pre_user_embed, pre_ad_embed, ad_feats, w2v_dict, [kf_index], max_len)
    sampler_test = PandasSample(pre_user_embed, pre_ad_embed, ad_feats, w2v_dict, [0], max_len)
    dataloader = DataLoader(
        dataset=sampler.get_seeds(),
        batch_size=batch_size,
        collate_fn=sampler.sampler,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers)
    dataloader_val = DataLoader(
        dataset=sampler_val.get_seeds(),
        batch_size=batch_size,
        collate_fn=sampler_val.sampler,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers)
    dataloader_test = DataLoader(
        dataset=sampler_test.get_seeds(),
        batch_size=batch_size,
        collate_fn=sampler_test.sampler,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers)

    # Define model and optimizer
    model = Net(
        dense_emb_N, dropout=dropout, max_len=max_len, user_embed_size=pre_num_hidden * 3,
        ad_embed_size=pre_num_hidden * 2, num_hidden=num_hidden
    )
    model = model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = th.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=num_epochs)
    # train
    avg = 0
    for epoch in range(num_epochs):
        tic = time.time()
        train(model, dataloader, device, epoch, lr_scheduler, label_smoothing, optimizer, log_every)
        toc = time.time()
        print('Epoch:{} Time(s): {:.4f}'.format(epoch + 1, toc - tic))
        if has_eval:
            eval(model, dataloader_val, device, epoch + 1)
        if epoch >= 5:
            avg += toc - tic
    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    if not has_eval:
        # 保存单模的预测结果，注意label输出要+1
        user_ids_val, pred_prob_val, label_val = predict(model, dataloader_val, device)
        user_ids_test, pred_prob_test, _ = predict(model, dataloader_test, device)
        id2vec = dict([(vocab_dict["user_id"][k], k) for k in vocab_dict["user_id"]])
        if kf_index == 1:
            pred_test = th.argmax(pred_prob_test, dim=1)
            pred_age_test = pred_test // 2
            pred_gender_test = pred_test % 2
            res_df = pd.DataFrame({
                "user_id": [id2vec[i] for i in user_ids_test.numpy().tolist()],
                "predicted_age": (pred_age_test + 1).numpy().tolist(),
                "predicted_gender": (pred_gender_test + 1).numpy().tolist(),
            })
            res_df.to_csv("submission_kf%s.csv" % kf_index, index=False)
        # 保存验针集到csv文件
        val_df = pd.DataFrame({
            "user_id": [id2vec[i] for i in user_ids_val.numpy().tolist()],
            "label": label_val.tolist()
        })
        for i in range(20):
            val_df["prob_%s" % i] = pred_prob_val[:, i].tolist()
        val_df.to_csv("val_kf_%s.csv" % kf_index, index=False)
        # 保存测试集
        test_df = pd.DataFrame({
            "user_id": [id2vec[i] for i in user_ids_test.numpy().tolist()],
        })
        for i in range(20):
            test_df["prob_%s" % i] = pred_prob_test[:, i].tolist()
        test_df.to_csv("test_kf_%s.csv" % kf_index, index=False)


if __name__ == '__main__':
    main()
