import multiprocessing

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

if __name__ == '__main__':
    # read csv
    ad_df = pd.read_csv("data/ad_i.csv")
    print("已加载：data/ad_i.csv")
    click_log_df = pd.read_csv("data/click_log_i.csv")
    print("已加载：data/click_log_i.csv")
    # 合并click_log和ad，这样可以方便的生成user_id和ad其他特征组成的图。
    click_log_df = click_log_df.merge(ad_df, on="creative_id")
    print("已完成merge：click_log_df")
    # 按用户和时间排序
    click_log_df = click_log_df.sort_values(["user_id", "time"])
    grouped = click_log_df.groupby("user_id")
    feat_list = [
        "creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry"
    ]
    for feat in feat_list:
        ad_list = []
        for name, group in tqdm(grouped):
            t = group[feat].to_list()
            if len(t) > 1:
                ad_list.append([str(i) for i in t])
        model = Word2Vec(
            ad_list, sg=1, size=128, window=8, seed=2020, min_count=1,
            workers=int(multiprocessing.cpu_count()) / 2, iter=10)
        embeddings = []
        for i in range(ad_df[feat].max() + 1):
            try:
                embeddings.append(model[str(i)])
            except:
                embeddings.append(np.zeros(128))
        embeddings = np.stack(embeddings)
        np.save("data/w2v_%s.npy" % feat, embeddings)

    # user_id
    grouped = click_log_df.groupby("creative_id")
    ad_list = []
    for name, group in tqdm(grouped):
        t = group["user_id"].to_list()
        if len(t) > 1:
            ad_list.append([str(i) for i in t])
    model = Word2Vec(
        ad_list, sg=1, size=128, window=8, seed=2020, min_count=1,
        workers=int(multiprocessing.cpu_count()) / 2, iter=10)
    embeddings = []
    for i in range(click_log_df["user_id"].max() + 1):
        try:
            embeddings.append(model[str(i)])
        except:
            embeddings.append(np.zeros(128))
    embeddings = np.stack(embeddings)
    np.save("data/w2v_%s.npy" % "user_id", embeddings)
