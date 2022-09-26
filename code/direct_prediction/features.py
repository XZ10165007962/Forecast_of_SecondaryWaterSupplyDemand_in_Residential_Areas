#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: features.py
@time: 2022/9/15 10:44
@version:
@desc: 
"""
import copy

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

import conf

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def get_features(all_data, lag=24, rolling=3):
	print("特征生成")
	features = []
	all_data, time_feature = time_features(all_data)
	features.extend(time_feature)
	# 获取滞后特征
	for i, j in enumerate(range(24*7, 24*7+lag+2)):
		all_data[f"flow_lag_{i}"] = all_data.groupby(["flow_id"])["flow"].shift(j)
		features.append(f"flow_lag_{i}")
		all_data[f"flow_lag_{i}_roll"] = all_data.groupby(["flow_id"])["flow"].shift(j).rolling(rolling).mean()
		features.append(f"flow_lag_{i}_roll")
		all_data[f"flow_lag_{i}_diff"] = all_data.groupby(["flow_id"])["flow"].shift(j).diff()
		features.append(f"flow_lag_{i}_diff")

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
	all_data["month_gep_pre"] = all_data["week_gep"] * all_data["flow_lag_0"]
	all_data["month_gep+pre"] = all_data["month_gep"] * all_data["flow_lag_0"]
	features.extend(["week_gep", "month_gep", "month_gep_pre", "month_gep+pre"])

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
	data_["is_weekday"] = list(map(lambda x: 1 if x >= 5 else 0, data_["dayofweek"]))
	data_["is_free"] = list(map(lambda x: 1 if x >= 7 and x <= 23 else 0, data_["hour"]))
	features = ["month", "day", "dayofweek", "hour", "is_weekday", "is_free"]
	return data_, features


if __name__ == '__main__':
	all_data_ = pd.read_csv(conf.tmp_data_paht + "all_data.csv")
	data, features = get_features(all_data_)
	data.to_csv(conf.tmp_data_paht + "feature_data.csv", index=False)