import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

tqdm.pandas(desc='pandas bar')

np.random.seed(1)

# 读取csv文件
click_log = pd.read_csv("data/click_log_i.csv")
ad = pd.read_csv("data/ad_i.csv")
train_user = pd.read_csv("data/user_i.csv")
# graph
user_num = len(set(click_log["user_id"].to_list()))
ad_num = len(set(click_log["creative_id"].to_list()))
# 边
click_log_t = click_log.groupby(by=["user_id", "creative_id"])["click_times"].agg("sum").reset_index()
row = click_log_t["user_id"].to_numpy()
col = click_log_t["creative_id"].to_numpy()
u2a_mat = np.stack([row, col])
click_times_mat = click_log_t["click_times"].to_numpy()
np.savez(
    "data/u2a_mat.npz",
    u2a_mat=u2a_mat,
    click_times_mat=click_times_mat
)

# id特征
age_arry = np.zeros((user_num,))
gender_arry = np.zeros((user_num,))
ad_id_arry = np.zeros((ad_num,))
product_id_arry = np.zeros((ad_num,))
product_category_arry = np.zeros((ad_num,))
advertiser_id_arry = np.zeros((ad_num,))
industry_arry = np.zeros((ad_num,))
for i in tqdm(ad.itertuples(), total=ad.shape[0]):
    ad_id_arry[i.creative_id] = i.ad_id
    product_id_arry[i.creative_id] = i.product_id
    product_category_arry[i.creative_id] = i.product_category
    advertiser_id_arry[i.creative_id] = i.advertiser_id
    industry_arry[i.creative_id] = i.industry
for i in tqdm(train_user.itertuples(), total=train_user.shape[0]):
    age_arry[i.user_id] = i.age
    gender_arry[i.user_id] = i.gender

# k折交叉验证。
train_mask = (age_arry != 0)
kf_index = np.zeros_like(age_arry)
train_index = np.where(train_mask)[0]
kf = KFold(n_splits=5, shuffle=True, random_state=2020)
for i, (train_ids, val_ids) in enumerate(kf.split(train_index)):
    kf_index[train_index[val_ids]] = i + 1

np.savez(
    "data/feature.npz",
    age=age_arry.astype(np.int32),
    gender=gender_arry.astype(np.int32),
    ad_id=ad_id_arry.astype(np.int32),
    product_id=product_id_arry.astype(np.int32),
    product_category=product_category_arry.astype(np.int32),
    advertiser_id=advertiser_id_arry.astype(np.int32),
    industry=industry_arry.astype(np.int32),
    kf_index=kf_index
)
