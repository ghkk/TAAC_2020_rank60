# encoding=utf8
import gc
import json
import os
import random
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from scipy import sparse
from scipy.special import iv
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as skshuffle
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings("ignore")


class ProNE():
    def __init__(self, graph_file, dimension):
        self.graph = graph_file
        self.dimension = dimension
        cache_matrix0_file = "data/matrix0.npz"
        if not os.path.exists(cache_matrix0_file):
            self.G = nx.read_edgelist(self.graph, nodetype=int, create_using=nx.DiGraph())
            self.G = self.G.to_undirected()
            self.node_number = self.G.number_of_nodes()
            matrix0 = scipy.sparse.lil_matrix((self.node_number, self.node_number))
            for e in self.G.edges():
                if e[0] != e[1]:
                    matrix0[e[0], e[1]] = 1
                    matrix0[e[1], e[0]] = 1
            self.matrix0 = scipy.sparse.csr_matrix(matrix0)
            sparse.save_npz(cache_matrix0_file, self.matrix0)
        else:
            self.matrix0 = sparse.load_npz(cache_matrix0_file)
            self.node_number = self.matrix0.shape[0]
        print(self.matrix0.shape)

    def get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]
        smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
        print('svd sparse', smat.data.shape[0] * 1.0 / l ** 2)
        U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        print('sparsesvd time', time.time() - t1)
        return U

    def get_embedding_dense(self, matrix, dimension):
        # get dense embedding via SVD
        t1 = time.time()
        U, s, Vh = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
        U = np.array(U)
        U = U[:, :dimension]
        s = s[:dimension]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        print('densesvd time', time.time() - t1)
        return U

    def pre_factorization(self, tran, mask):
        # Network Embedding as Sparse Matrix Factorization
        t1 = time.time()
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1
        neg = neg / neg.sum()
        neg = scipy.sparse.diags(neg, format="csr")
        neg = mask.dot(neg)
        print("neg", time.time() - t1)
        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1
        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)
        C1 -= neg
        F = C1
        features_matrix = self.get_embedding_rand(F)
        return features_matrix

    def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        # NE Enhancement via Spectral Propagation
        print('Chebyshev Series -----------------')
        t1 = time.time()
        if order == 1:
            return a
        A = sp.eye(self.node_number) + A
        DA = preprocessing.normalize(A, norm='l1')
        L = sp.eye(self.node_number) - DA
        M = L - mu * sp.eye(self.node_number)
        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a
        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            #         Lx2 = 2*L.dot(Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
            print('Bessell time', i, time.time() - t1)
        mm = A.dot(a - conv)
        emb = self.get_embedding_dense(mm, self.dimension)
        return emb


def evaluate(user_df, features_matrix):
    features_matrix = features_matrix[user_df["user_id"].to_list(),]
    print(features_matrix.shape)
    nodesize = features_matrix.shape[0]
    label_matrix = user_df["age"]
    label_matrix = label_matrix.to_numpy()
    label_matrix = np.stack([label_matrix]).T - 1
    train_percent = 0.7

    random.seed(1)
    np.random.seed(1)
    res = []
    for i in range(4):
        t_1 = time.time()
        X, y = skshuffle(features_matrix, label_matrix)
        training_size = int(train_percent * nodesize)
        X_train = X[:training_size, :]
        y_train = y[:training_size, :]
        X_test = X[training_size:, :]
        y_test = y[training_size:, :]

        clf = LogisticRegression(random_state=0, solver="saga", multi_class="multinomial")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = (preds == y_test[:, 0]).sum() / len(y_test)
        res.append(acc)
        print(time.time() - t_1, "s")
    print("avg age acc:", sum(res) / len(res))
    print("min age acc:", min(res))
    print("max age acc:", max(res))


if __name__ == '__main__':
    step = 10
    theta = 0.5
    graph = "data/edgelist_tencent.ungraph"

    # read csv
    user_df = pd.read_csv("data/user_i.csv")
    print("已加载：data/user_i.csv")
    ad_df = pd.read_csv("data/ad_i.csv")
    print("已加载：data/ad_i.csv")
    click_log_df = pd.read_csv("data/click_log_i.csv")
    click_log_df = click_log_df[["user_id", "creative_id"]].drop_duplicates(
        ["user_id", "creative_id"])
    print("已加载：data/click_log_i.csv")
    # read vocab
    with open("data/vocab.json") as f:
        vocab = json.load(f)
    print("已加载：data/vocab.json")
    # id的顺序是
    # user_id,
    # creative_id, ad_id, product_id, product_category, advertiser_id, industry
    feat_list = [
        "user_id",
        "creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry"
    ]
    n = 0
    feat_count = []
    for feat_name in feat_list:
        feat_count.append(n)
        n += len(vocab[feat_name])
        if feat_name != "user_id":
            ad_df[feat_name] += feat_count[-1]
        if feat_name == "creative_id":
            click_log_df[feat_name] += feat_count[-1]
    print("已将所有属性id拼到一个连续空间")
    # 生成图文件
    if not os.path.exists(graph):
        with open(graph, "w") as f:
            # 遍历点击记录
            for i in click_log_df.itertuples():
                f.write("%s\t%s\n" % (i.user_id, i.creative_id))
            # 遍历广告表
            for i in ad_df.itertuples():
                f.write("%s\t%s\n" % (i.creative_id, i.ad_id))
                f.write("%s\t%s\n" % (i.creative_id, i.product_id))
                f.write("%s\t%s\n" % (i.creative_id, i.product_category))
                f.write("%s\t%s\n" % (i.creative_id, i.advertiser_id))
                f.write("%s\t%s\n" % (i.creative_id, i.industry))
    print("已经生成图")

    # 训练128维
    dims = 128

    t_0 = time.time()
    model = ProNE(graph, dims)
    t_1 = time.time()
    features_matrix_file = "data/features_matrix_%s.npy" % dims
    if os.path.exists(features_matrix_file):
        features_matrix = np.load(features_matrix_file)
    else:
        features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
        np.save(features_matrix_file, features_matrix)
    t_2 = time.time()
    print('---', model.node_number)
    print('sparse NE time', t_2 - t_1)
    t_3 = t_2
    # 在0附近搜索mu参数可以获得更好的结果。例如，增强实验中Wikipedia上的mu = -4.0（低通）时的结果要比论文中报道的结果更好。
    for mu in [-1.0]:
        gc.collect()
        mu = round(mu, 2)
        embed_file = "data/tencent_%s_%s.npy" % (dims, mu)
        print("#" * 20)
        print(embed_file)
        t_2 = t_3
        if os.path.exists(embed_file):
            embeddings_matrix = np.load(embed_file)
        else:
            embeddings_matrix = model.chebyshev_gaussian(model.matrix0, features_matrix, step, mu, theta)
            np.save(embed_file, embeddings_matrix)
        t_3 = time.time()
        print('spectral Pro time', t_3 - t_2)
    #         # 评估
    #         evaluate(user_df, embeddings_matrix)

    # 训练16维
    dims = 16

    t_0 = time.time()
    model = ProNE(graph, dims)
    t_1 = time.time()
    features_matrix_file = "data/features_matrix_%s.npy" % dims
    if os.path.exists(features_matrix_file):
        features_matrix = np.load(features_matrix_file)
    else:
        features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
        np.save(features_matrix_file, features_matrix)
    t_2 = time.time()
    print('---', model.node_number)
    print('sparse NE time', t_2 - t_1)
    t_3 = t_2
    # 在0附近搜索mu参数可以获得更好的结果。例如，增强实验中Wikipedia上的mu = -4.0（低通）时的结果要比论文中报道的结果更好。
    for mu in [-0.6]:
        gc.collect()
        mu = round(mu, 2)
        embed_file = "data/tencent_%s_%s.npy" % (dims, mu)
        print("#" * 20)
        print(embed_file)
        t_2 = t_3
        if os.path.exists(embed_file):
            embeddings_matrix = np.load(embed_file)
        else:
            embeddings_matrix = model.chebyshev_gaussian(model.matrix0, features_matrix, step, mu, theta)
            np.save(embed_file, embeddings_matrix)
        t_3 = time.time()
        print('spectral Pro time', t_3 - t_2)
#         # 评估
#         evaluate(user_df, embeddings_matrix)
