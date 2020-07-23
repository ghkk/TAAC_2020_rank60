import json

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas(desc='pandas bar')

np.random.seed(1)

# 合并数据集
train_ad = pd.read_csv("data/train_preliminary/ad.csv")
train_click_log = pd.read_csv("data/train_preliminary/click_log.csv")
train_user = pd.read_csv("data/train_preliminary/user.csv")

train_ad2 = pd.read_csv("data/train_semi_final/ad.csv")
train_click_log2 = pd.read_csv("data/train_semi_final/click_log.csv")
train_user2 = pd.read_csv("data/train_semi_final/user.csv")

test_ad = pd.read_csv("data/test/ad.csv")
test_click_log = pd.read_csv("data/test/click_log.csv")

# ad: creative_id、ad_id、product_id、product_category、advertiser_id、industry。其中creative_id是唯一id。3412772个样本。
ad = pd.concat([train_ad, train_ad2, test_ad]) \
    .drop_duplicates(["creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry"])
# click_log: time、user_id、creative_id、click_times。user_id、creative_id都是唯一id，没有重复。63668283个样本。
click_log = pd.concat([train_click_log, train_click_log2, test_click_log]) \
    .drop_duplicates(["time", "user_id", "creative_id", "click_times"])
# train_user：user_id、age、gender。user_id是唯一id。1900000个样本，其中训练集900000个样本，测试集1000000个样本。
train_user = pd.concat([train_user, train_user2]) \
    .drop_duplicates(["user_id", "age", "gender"])

# 保存新数据
ad.to_csv("data/ad.csv", index=None)
click_log.to_csv("data/click_log.csv", index=None)
train_user.to_csv("data/user.csv", index=None)

# 读取csv文件
click_log = pd.read_csv("data/click_log.csv")
click_log["time"] = click_log["time"].progress_apply(str)
click_log["user_id"] = click_log["user_id"].progress_apply(str)
click_log["creative_id"] = click_log["creative_id"].progress_apply(str)
ad = pd.read_csv("data/ad.csv")
ad["creative_id"] = ad["creative_id"].progress_apply(str)
ad["ad_id"] = ad["ad_id"].progress_apply(str)
ad["product_id"] = ad["product_id"].progress_apply(str)
ad["product_category"] = ad["product_category"].progress_apply(str)
ad["advertiser_id"] = ad["advertiser_id"].progress_apply(str)
ad["industry"] = ad["industry"].progress_apply(str)
train_user = pd.read_csv("data/user.csv")
train_user["user_id"] = train_user["user_id"].progress_apply(str)

# 由于这些id不都是连续的，因此需要重新映射。
# 由于大概率所有数据都有机会参与图的运算，且不存在推理时的新item，因此不设置<unkown>标签了。可能需要加与频率相关的正则了。
# 统计每个特征出现的次数。先统计click_log中time、user_id、creative_id的出现次数。
freq_dict = {}
freq_dict["time"] = {}
freq_dict["user_id"] = {}
freq_dict["creative_id"] = {}
for i in tqdm(click_log.itertuples(), total=click_log.shape[0]):
    if i.time not in freq_dict["time"]:
        freq_dict["time"][i.time] = 0
    freq_dict["time"][i.time] += 1
    if i.user_id not in freq_dict["user_id"]:
        freq_dict["user_id"][i.user_id] = 0
    freq_dict["user_id"][i.user_id] += 1
    if i.creative_id not in freq_dict["creative_id"]:
        freq_dict["creative_id"][i.creative_id] = 0
    freq_dict["creative_id"][i.creative_id] += 1
# 再用这些次数对ad的ad_id、product_id、product_category、advertiser_id、industry进行加权统计。
# 这是在已知所有id都在click_log中出现这个前提下的。
freq_dict["ad_id"] = {}
freq_dict["product_id"] = {}
freq_dict["product_category"] = {}
freq_dict["advertiser_id"] = {}
freq_dict["industry"] = {}
for i in tqdm(ad.itertuples(), total=ad.shape[0]):
    if i.ad_id not in freq_dict["ad_id"]:
        freq_dict["ad_id"][i.ad_id] = 0
    freq_dict["ad_id"][i.ad_id] += freq_dict["creative_id"][i.creative_id]
    if i.product_id not in freq_dict["product_id"]:
        freq_dict["product_id"][i.product_id] = 0
    freq_dict["product_id"][i.product_id] += freq_dict["creative_id"][i.creative_id]
    if i.product_category not in freq_dict["product_category"]:
        freq_dict["product_category"][i.product_category] = 0
    freq_dict["product_category"][i.product_category] += freq_dict["creative_id"][i.creative_id]
    if i.advertiser_id not in freq_dict["advertiser_id"]:
        freq_dict["advertiser_id"][i.advertiser_id] = 0
    freq_dict["advertiser_id"][i.advertiser_id] += freq_dict["creative_id"][i.creative_id]
    if i.industry not in freq_dict["industry"]:
        freq_dict["industry"][i.industry] = 0
    freq_dict["industry"][i.industry] += freq_dict["creative_id"][i.creative_id]
# 构建字典和词频文件
vocab_dict = {}
for feature_name in tqdm(freq_dict, desc="parse vocab"):
    f_freq = freq_dict[feature_name]
    f_freq = sorted([(k, f_freq[k]) for k in f_freq], key=lambda x: -x[1])
    vocab_dict[feature_name] = dict([(k, i) for i, (k, v) in enumerate(f_freq)])
    freq_dict[feature_name] = [v for k, v in f_freq]

with open("data/vocab.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(vocab_dict, ensure_ascii=False, indent=4))
with open("data/freq.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(freq_dict, ensure_ascii=False, indent=4))

# 映射csv文件并保存
ad_df = pd.read_csv("data/ad.csv")
print("已加载：data/ad.csv")
user_df = pd.read_csv("data/user.csv")
print("已加载：data/user.csv")
click_log_df = pd.read_csv("data/click_log.csv")
print("已加载：data/click_log.csv")
# read vocab
with open("data/vocab.json") as f:
    vocab = json.load(f)
print("已加载：data/vocab.json")
# id2index
for i, k in enumerate(ad_df.columns):
    ad_df[k] = ad_df.progress_apply(lambda x: vocab[k][str(x[i])], axis=1)
print("已完成映射：ad_df")
for i, k in enumerate(user_df.columns):
    if k in ["user_id"]:
        user_df[k] = user_df.progress_apply(lambda x: vocab[k][str(x[i])], axis=1)
print("已完成映射：user_df")
for i, k in enumerate(click_log_df.columns):
    if k in ["time", "user_id", "creative_id"]:
        click_log_df[k] = click_log_df.progress_apply(lambda x: vocab[k][str(x[i])], axis=1)
print("已完成映射：click_log_df")
ad_df.to_csv("data/ad_i.csv", index=None)
user_df.to_csv("data/user_i.csv", index=None)
click_log_df.to_csv("data/click_log_i.csv", index=None)
print("已保存映射后csv")
