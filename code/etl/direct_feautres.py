#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: direct_feautres.py
@time: 2022/10/8 14:25
@version:
@desc: 
"""
import copy

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

from code.util import conf

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

def get_features(all_data, lag=36, rolling=3):
	print("特征生成")
	features = []
	all_data, time_feature = time_features(all_data)
	features.extend(time_feature)
	# 获取滞后特征
	for i, j in enumerate(range(24*7, 24*7+lag)):
		all_data[f"flow_lag_{i}"] = all_data.groupby(["flow_id"])["flow"].shift(j)
		features.append(f"flow_lag_{i}")
		all_data[f"flow_lag_{i}_roll"] = all_data.groupby(["flow_id"])["flow"].shift(j).rolling(rolling).mean()
		features.append(f"flow_lag_{i}_roll")
		# all_data[f"flow_lag_{i}_diff"] = all_data.groupby(["flow_id"])["flow"].shift(j).diff()
		# features.append(f"flow_lag_{i}_diff")
		# all_data[f"flow_lag_{i}_diff_lag"] = all_data[f"flow_lag_{i}"] + all_data[f"flow_lag_{i}_diff"]
		# features.append(f"flow_lag_{i}_diff_lag")
	# 同比
	week_lag = 3
	for i in range(7, week_lag + 8):
		if i >= 2:
			all_data[f"flow_lag_{24 * i - 1}"] = all_data.groupby(["flow_id"])["flow"].shift(24 * i - 1)
			all_data[f"flow_lag_{24 * i}"] = all_data.groupby(["flow_id"])["flow"].shift(24 * i)
			all_data[f"flow_lag_{24 * i + 1}"] = all_data.groupby(["flow_id"])["flow"].shift(24 * i + 1)
			features.append(f"flow_lag_{24 * i - 1}")
			features.append(f"flow_lag_{24 * i}")
			features.append(f"flow_lag_{24 * i + 1}")

			all_data[f"flow_lag_{24 * i - 1}_{24 * i}_{24 * i + 1}_mean"] = \
				all_data.loc[:, [f"flow_lag_{24 * i - 1}", f"flow_lag_{24 * i}", f"flow_lag_{24 * i + 1}"]].mean(axis=1)
			features.append(f"flow_lag_{24 * i - 1}_{24 * i}_{24 * i + 1}_mean")
			all_data[f"flow_lag_{24 * i}_{24 * i + 1}_mean"] = \
				all_data.loc[:, [f"flow_lag_{24 * i}", f"flow_lag_{24 * i + 1}"]].mean(axis=1)
			features.append(f"flow_lag_{24 * i}_{24 * i + 1}_mean")

	tongbi_col = [f"flow_lag_{24 * i}" for i in range(7, week_lag + 8)]
	funcs = ["mean", "sum", "median", "max", "min", "std"]
	for func in funcs:
		all_data[f"tongbi_{func}"] = all_data[tongbi_col].apply(func, axis=1)
		features.append(f"tongbi_{func}")

	# 计算统计特征
	funcs = ["mean", "sum", "median", "max", "min", "std"]
	flag = 3
	for func in funcs:
		all_data[f"flow_id_{func}"] = all_data.groupby(["flow_id"])["flow"].transform(func)
		features.append(f"flow_id_{func}")
	for func in funcs:
		simple = copy.deepcopy(all_data)
		simple[f"month_{func}"] = simple.groupby(["month", "flow_id"])["flow"].transform(func)
		simple[f"month_weekday_{func}"] = simple.groupby(["month", "is_weekday", "flow_id"])["flow"].transform(func)
		for i in range(1, flag):
			simple[f"month_{func}_{i}"] = simple[f"month_{func}"]
			simple[f"month_weekday_{func}_{i}"] = simple[f"month_weekday_{func}"]
			features.append(f"month_{func}_{i}")
			features.append(f"month_weekday_{func}_{i}")
			simple["month"] = list(map(lambda x: x+i, simple["month"]))
			all_data = all_data.merge(simple.loc[:, ["month", "flow_id", f"month_{func}_{i}"]].drop_duplicates(),
									  on=["month", "flow_id"], how="left")
			all_data = all_data.merge(simple.loc[:, ["month", "is_weekday", "flow_id", f"month_weekday_{func}_{i}"]].drop_duplicates(),
									  on=["month", "is_weekday", "flow_id"], how="left")
	for func in funcs:
		simple = copy.deepcopy(all_data)
		simple[f"week_{func}"] = simple.groupby(["week", "flow_id"])["flow"].transform(func)
		simple[f"week_free_{func}"] = simple.groupby(["week", "is_free", "flow_id"])["flow"].transform(func)
		simple[f"week_weekday_{func}"] = simple.groupby(["week", "is_weekday", "flow_id"])["flow"].transform(func)
		for i in range(1, flag):
			simple[f"week_{func}_{i}"] = simple[f"week_{func}"]
			simple[f"week_free_{func}_{i}"] = simple[f"week_free_{func}"]
			simple[f"week_weekday_{func}_{i}"] = simple[f"week_weekday_{func}"]
			features.append(f"week_{func}_{i}")
			features.append(f"week_free_{func}_{i}")
			features.append(f"week_weekday_{func}_{i}")
			simple["week"] = list(map(lambda x: x+i, simple["week"]))
			all_data = all_data.merge(simple.loc[:, ["week", "flow_id", f"week_{func}_{i}"]].drop_duplicates(),
									  on=["week", "flow_id"], how="left")
			all_data = all_data.merge(simple.loc[:, ["week", "flow_id", "is_free", f"week_free_{func}_{i}"]].drop_duplicates(),
									  on=["week", "is_free", "flow_id"], how="left")
			all_data = all_data.merge(
				simple.loc[:, ["week", "flow_id", "is_weekday", f"week_weekday_{func}_{i}"]].drop_duplicates(),
				on=["week", "is_weekday", "flow_id"], how="left")
	all_data["week_gep"] = all_data["week_mean_1"] / all_data["week_mean_2"]
	all_data["month_gep"] = all_data["month_mean_1"] / all_data["month_mean_2"]
	all_data["week_gep_pre"] = all_data["week_gep"] * all_data["flow_lag_0"]
	all_data["month_gep_pre"] = all_data["month_gep"] * all_data["flow_lag_0"]
	features.extend(["week_gep", "month_gep", "week_gep_pre", "month_gep_pre"])
	cats = ["is_weekday", "is_free"]
	for cat in cats:
		all_data[cat] = all_data[cat].astype('category')

	return all_data, features, cats


def time_features(data_):
	data_["time"] = pd.to_datetime(data_["time"])
	data_["month"] = data_["time"].dt.month
	data_["week"] = data_["time"].dt.week
	data_["day"] = data_["time"].dt.day
	data_["dayofweek"] = data_["time"].dt.dayofweek
	data_["dayofyear"] = data_["time"].dt.dayofyear
	data_["hour"] = data_["time"].dt.hour
	data_['sin_hour'] = np.sin(2 * np.pi * data_["hour"] / 24)
	data_['cos_hour'] = np.cos(2 * np.pi * data_["hour"] / 24)
	data_["is_weekday"] = list(map(lambda x: 1 if x >= 5 else 0, data_["dayofweek"]))
	data_["is_free"] = list(map(lambda x: 1 if x >= 7 and x <= 23 else 0, data_["hour"]))
	features = ["month", "day", "dayofweek", "hour", "is_weekday", "is_free"]
	# jiaqibiao = [1, 2, 3, 31, 32, 33, 34, 35, 36, 37, 93, 94, 95, 120, 121, 122, 123, 124, 154, 155, 156]
	# jiaqibiao_after = [30, 92, 119, 153]
	# jiaqibiao_latter = [4, 38, 96, 125, 157]
	# data_["jiaqi"] = list(map(lambda x: 1 if x in jiaqibiao else 0, data_["dayofyear"]))
	# data_["jiaqi_after"] = list(map(lambda x: 1 if x in jiaqibiao_after else 0, data_["dayofyear"]))
	# data_["jiaqi_latter"] = list(map(lambda x: 1 if x in jiaqibiao_latter else 0, data_["dayofyear"]))
	# features.extend(["jiaqi", "jiaqi_after", "jiaqi_latter"])
	return data_, features


if __name__ == '__main__':
	all_data_ = pd.read_csv(conf.tmp_data_paht + "all_data.csv")
	data, features, _ = get_features(all_data_)
	data.dropna(inplace=True)
	print(data.loc[83126, :])
	data.to_csv(conf.tmp_data_paht + "feature_data.csv", index=False)
	# all_data_ = pd.read_csv(conf.tmp_data_paht + "feature_data.csv")
	# print(all_data_.corr())