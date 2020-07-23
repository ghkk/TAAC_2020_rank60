import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas(desc='pandas bar')

np.random.seed(1)


def clip_max_len(vec, max_len):
    if len(vec) > max_len:
        vec = vec[:max_len]
    elif len(vec) < max_len:
        vec = vec + [0 for i in range(max_len - len(vec))]
    return vec


# 读取csv文件
print("读取交互数据。。。")
click_log = pd.read_csv("data/click_log_i.csv")
user_num = len(set(click_log["user_id"].to_list()))
ad_num = len(set(click_log["creative_id"].to_list()))
# 不考虑顺序，则特征为：用户id、广告id、广告点击量、广告tf-idf
print("开始按无序的方式处理。。。")
click_log_t = click_log.groupby(by=["user_id", "creative_id"])["click_times"].agg("sum").reset_index()
feats = np.load("data/feature.npz")
age = feats["age"]
gender = feats["gender"]
kf_index = feats["kf_index"]
df_user = pd.DataFrame({"user_id": np.arange(user_num), "age": age, "gender": gender, "kf_index": kf_index})
click_log_t = click_log_t.merge(df_user)
## 统计广告idf和每个用户总点击次数
print("统计TFIDF。。。")
idf = np.log(user_num / click_log_t.groupby(by="creative_id")["user_id"].agg("count").reset_index()["user_id"] + 1)
count = click_log_t.groupby(by="user_id")["click_times"].agg("sum").reset_index()["click_times"]
click_log_t = click_log_t.merge(pd.DataFrame({"user_id": np.arange(user_num), "count": count}))
click_log_t = click_log_t.merge(pd.DataFrame({"creative_id": np.arange(ad_num), "idf": idf}))
click_log_t["tfidf"] = click_log_t["click_times"] / click_log_t["count"] * click_log_t["idf"]
print("收集用户的交互序列。。。")


def foo(df_i):
    return pd.DataFrame(
        {
            'creative_id': [list(df_i["creative_id"])],
            'click_times': [list(df_i["click_times"])],
            'tfidf': [list(df_i["tfidf"])],
            'age': [df_i["age"].to_list()[0]],
            'gender': [df_i["gender"].to_list()[0]],
            'kf_index': [df_i["kf_index"].to_list()[0]]
        }
    )


df = click_log_t[["user_id", "creative_id", "click_times", "age", "gender", "kf_index", "tfidf"]].groupby(
    'user_id').apply(foo).reset_index()
for kf in range(6):
    df_i = df[df["kf_index"] == kf]
    print("保存第{}个数据集。。。".format(kf))
    df_i.to_pickle("data/data_kf_{}.pkl".format(kf))

# 考虑顺序，则特征为：用户id、广告id、广告点击量、广告tf-idf、点击时间
print("开始按有序的方式处理。。。")
click_log = click_log.merge(df_user)
click_log = click_log.merge(pd.DataFrame({"user_id": np.arange(user_num), "count": count}))
click_log = click_log.merge(pd.DataFrame({"creative_id": np.arange(ad_num), "idf": idf}))
click_log["tfidf"] = click_log["click_times"] / click_log["count"] * click_log["idf"]
click_log = click_log.sort_values(["user_id", "time"], ascending=False)
print("收集用户的交互序列。。。")


def foo(df_i):
    return pd.DataFrame(
        {
            'creative_id': [list(df_i["creative_id"])],
            'click_times': [list(df_i["click_times"])],
            'tfidf': [list(df_i["tfidf"])],
            'time': [list(df_i["time"])],
            'age': [df_i["age"].to_list()[0]],
            'gender': [df_i["gender"].to_list()[0]],
            'kf_index': [df_i["kf_index"].to_list()[0]]
        }
    )


df = click_log[["user_id", "creative_id", "click_times", "age", "gender", "kf_index", "tfidf", "time"]].groupby(
    'user_id').apply(foo).reset_index()
for kf in range(6):
    df_i = df[df["kf_index"] == kf]
    print("保存第{}个数据集。。。".format(kf))
    df_i.to_pickle("data/data_time_kf_{}.pkl".format(kf))
