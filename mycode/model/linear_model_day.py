#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: linear_model.py
@time: 2022/10/8 15:08
@version:
@desc:
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt    #画图的

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR,LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mycode.util import conf
from mycode.util.util import MSLE

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def fueat(data, day):
	feat_col = []
	use_col = []
	for i in range(day, day+8):
		data[f"flow_{i}"] = data.groupby(["flow_id"])["flow"].shift(24*i)
		feat_col.extend([f"flow_{i}"])
		use_col.extend([f"flow_{i}"])
		for func in ["mean", "max", "min"]:
			if func == "mean":
				data[f"flow_{i}_roll_{func}"] = data.groupby(["flow_id"])["flow"].shift(24 * i).rolling(6).mean()
			if func == "max":
				data[f"flow_{i}_roll_{func}"] = data.groupby(["flow_id"])["flow"].shift(24 * i).rolling(6).max()
			if func == "min":
				data[f"flow_{i}_roll_{func}"] = data.groupby(["flow_id"])["flow"].shift(24 * i).rolling(6).min()
			feat_col.extend([f"flow_{i}_roll_{func}"])
	data["flow_mean"] = data[use_col].mean(axis=1)
	data["flow_max"] = data[use_col].max(axis=1)
	data["flow_min"] = data[use_col].min(axis=1)
	data["flow_sum"] = data[use_col].sum(axis=1)
	data["flow_std"] = data[use_col].std(axis=1)
	feat_col.extend(["flow_mean", "flow_max", "flow_min", "flow_sum", "flow_std"])
	for i in range(24*day, 24*day+13):
		data[f"flow_lag_{i}"] = data.groupby(["flow_id"])["flow"].shift(i)
		data[f"flow_roll_{i}"] = data.groupby(["flow_id"])["flow"].shift(i).rolling(i+1).mean()
		feat_col.extend([f"flow_lag_{i}", f"flow_roll_{i}"])
	data["time"] = pd.to_datetime(data["time"])
	data["month"] = data["time"].dt.month
	data["dayofweek"] = data["time"].dt.dayofweek
	data["hour"] = data["time"].dt.hour
	data['sin_hour'] = np.sin(2 * np.pi * data["hour"] / 24)
	data['cos_hour'] = np.cos(2 * np.pi * data["hour"] / 24)
	feat_col.extend(["month", "dayofweek","hour",'sin_hour','cos_hour'])
	data["month_mean"] = data.groupby(["flow_id", "month"])["flow"].transform("mean")
	data["dayofweek_mean"] = data.groupby(["flow_id", "dayofweek"])["flow"].transform("mean")
	data["hour_mean"] = data.groupby(["flow_id", "hour"])["flow"].transform("mean")
	feat_col.extend(["hour_mean"])
	return data, feat_col


def model(X_train, X_test, y_train, y_test, flag=0, if_train=True):
	if flag == 0:
		model_list = {"knn": KNeighborsRegressor(weights="distance",algorithm="ball_tree",n_neighbors=3),
					  "svr": SVR(),
					  "bagging": BaggingRegressor(random_state=2022, n_estimators=100, max_samples=0.9,
												  max_features=0.9),
					  "randomtree": RandomForestRegressor(random_state=2022),
					  "linear": LinearRegression()
					  }
		# knn， svr， bagg， 随机森林
		if if_train:
			pre_list_all = np.zeros((len(X_test),))
			pre_list_knn = []
			pre_list_svr = []
			pre_list_bagg = []
			pre_list_rand = []
			pre_list_linear = []
			for key, model in model_list.items():
				print(f"============{key}========")
				# n_splits = 5
				# pre = 0
				# kf = KFold(n_splits=n_splits)
				# for tra_index, val_index in kf.split(X_train):
				# 	model.fit(X_train[tra_index], y_train[tra_index])
				# 	pre += model.predict(X_test) / n_splits
				model.fit(X_train, y_train)
				pre = model.predict(X_test)
				if key == "knn":
					pre_list_knn.extend(pre)
				elif key == "svr":
					pre_list_svr.extend(pre)
				elif key == "bagging":
					pre_list_bagg.extend(pre)
				elif key == "randomtree":
					pre_list_rand.extend(pre)
				else:
					pre_list_linear.extend(pre)
				pre_list_all += pre / len(model_list)
		return pre_list_all, pre_list_knn, pre_list_svr, pre_list_bagg, pre_list_rand, pre_list_linear
	if flag == 1:
		estimators = [
			('knn', KNeighborsRegressor()),
			('svr', SVR()),
			('bagging', BaggingRegressor(random_state=2022, n_estimators=100, max_samples=0.9, max_features=0.9)),
			('randomtree', RandomForestRegressor(random_state=2022))
		]
		reg = StackingRegressor(
			estimators=estimators,
			final_estimator=LinearRegression(),
			cv=5
		)
		if if_train:
			reg.fit(X_train, y_train)
			pre = reg.predict(X_test)
			return pre


def train(data, flag = 0):
	train1 = data[(data['time'] >= '2022-01-01 01:00:00') & (data['time'] < '2022-05-01 01:00:00')].reset_index(
		drop=True)
	train2 = data[(data['time'] >= '2022-05-08 01:00:00') & (data['time'] < '2022-06-01 01:00:00')].reset_index(
		drop=True)
	train3 = data[(data['time'] >= '2022-06-08 01:00:00') & (data['time'] < '2022-07-21 01:00:00')].reset_index(
		drop=True)
	train4 = data[(data['time'] >= '2022-07-28 01:00:00') & (data['time'] < '2022-08-21 01:00:00')].reset_index(
		drop=True)
	train1["pre"] = 0
	pre_list = []
	pre_list_knn = []
	pre_list_svr = []
	pre_list_bagg = []
	pre_list_rand = []
	pre_list_linear = []
	test_list = []

	for id in [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]:
		print(f"+++++++++++++++++{id}+++++++++++++++++++++")
		for day in range(7, 0, -1):
			data, feat = fueat(train1, day)
			train_ = data[data['time'] >= '2022-03-01 01:00:00']
			train__ = train_[train_["flow_id"] == id]
			print(f"+++++++++++++++++{day}+++++++++++++++++++++")
			train = train__.iloc[:-24*day, :]
			X_data, y_data = train.loc[:, feat], train.loc[:, ["flow"]].values
			stand = MinMaxScaler()
			X_data = stand.fit_transform(X_data)
			X_train, y_train = X_data[:-24*(day+1)], y_data[:-24*(day+1)].flatten()
			X_test, y_test = X_data[-24*(day+1):], y_data[-24*(day+1):].flatten()
			if flag:
				pre = model(X_train, X_test, y_train, y_test, flag=flag, if_train=True)
				pre_list.extend(pre)
				test_list.extend(y_test)
				print("mlse:", MSLE(test_list, pre_list, flag=1))
				train1.loc[train__.iloc[-24 * (day + 1):, :].index, ["flow"]] = pre
			else:
				pre_all, pre_knn, pre_svr, pre_bagg, pre_rand, pr_linear = model(X_train, X_test, y_train, y_test, flag=flag, if_train=True)
				pre_list.extend(pre_all)
				pre_list_knn.extend(pre_knn)
				pre_list_svr.extend(pre_svr)
				pre_list_bagg.extend(pre_bagg)
				pre_list_rand.extend(pre_rand)
				pre_list_linear.extend(pr_linear)
				test_list.extend(y_test)
				print("knn mlse:", MSLE(test_list, pre_list_knn, flag=1))
				print("svr mlse:", MSLE(test_list, pre_list_svr, flag=1))
				print("bagg mlse:", MSLE(test_list, pre_list_bagg, flag=1))
				print("rand mlse:", MSLE(test_list, pre_list_rand, flag=1))
				print("linear mlse:", MSLE(test_list, pre_list_linear, flag=1))
				print("mlse:", MSLE(test_list, pre_list, flag=1))
				train1.loc[train__.iloc[-24*(day+1):, :].index, ["flow"]] = pre_all


def model_test(data):
	orinal_data = data
	orinal_data["knn"] = 0
	orinal_data["svr"] = 0
	orinal_data["bagging"] = 0
	orinal_data["randomtree"] = 0
	model_list = {"knn": KNeighborsRegressor(weights="distance",algorithm="ball_tree",n_neighbors=3),
				  "svr": SVR(),
				  "bagging": BaggingRegressor(random_state=2022, n_estimators=100, max_samples=0.9, max_features=0.9),
				  "randomtree": RandomForestRegressor(random_state=2022),
				  "linear": LinearRegression()
				  }
	test_list = ["test1", "test2", "test3", "test4"]
	estimators = [
		('knn', KNeighborsRegressor()),
		('svr', SVR()),
		('bagging', BaggingRegressor(random_state=2022, n_estimators=100, max_samples=0.9, max_features=0.9)),
		('randomtree', RandomForestRegressor(random_state=2022))
	]
	reg = StackingRegressor(
		estimators=estimators,
		final_estimator=LinearRegression(),
		cv=5
	)
	for i, time in enumerate(['2022-05-08 00:00:00', '2022-06-08 00:00:00', '2022-07-28 00:00:00', '2022-08-28 00:00:00']):

		data_ = orinal_data[orinal_data["time"] <= time]
		for id in [
			"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10",
			"flow_11",
			"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
		]:
			pre_final = []
			print(f"+++++++++++++++++{id}+++++++++++++++++++++")
			id_train = data_[data_["flow_id"] == id]
			for day in range(7, 0, -1):
				feat_data, feat = fueat(id_train, day)
				train_ = feat_data[feat_data['time'] >= '2022-02-15 01:00:00']
				if day == 1:
					train = train_
				else:
					train = train_.iloc[:-24 * (day-1), :]
				# print(train_.iloc[-24:])
				X_data, y_data = train.loc[:, feat], train.loc[:, ["flow"]].values
				# 对其进行标准化
				stand = MinMaxScaler()
				X_data = stand.fit_transform(X_data)
				X_train, y_train = X_data[:-24 * day], y_data[:-24 * day].flatten()
				X_test = X_data[-24:]
				print(f"+++++++++++++++++{day}+++++++++++++++++++++")
				# print(id_train.iloc[-24 * day:-24 * (day-1)])
				pre_ser = np.zeros((24,))  # 存储预测结果
				for key, model in model_list.items():
					model.fit(X_train, y_train)
					pre = model.predict(X_test)
					pre_ser += pre/len(model_list)
					if day != 1:
						id_train.loc[id_train.iloc[-24 * day:-24 * (day-1), :].index, key] = pre
						id_train.loc[id_train.iloc[-24 * day:-24 * (day - 1), :].index, "flow"] = pre_ser
				pre_final.extend(pre_ser.tolist())
			# reg.fit(X_train, y_train)
			# pre = reg.predict(X_test)
			# pre_final.extend(pre)
			# 将预测结果拼接回原始数据
			orinal_data.loc[id_train.iloc[-24 * 7:, :].index, "flow"] = pre_final
	orinal_data.to_csv(conf.predict_data_path + "orinal_data.csv", index=False)
	# 保存提交结果
	all_data = orinal_data[orinal_data["train or test"] != "train"]
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10",
		"flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	sub = pd.DataFrame()
	for i, flow in enumerate(flow_id):
		temp = all_data[all_data["flow_id"] == flow].reset_index(drop=True)
		if i == 0:
			sub = temp.loc[:, ["time", "flow"]]
		else:
			sub = pd.concat([sub, temp.loc[:, ["flow"]]], axis=1)
	sub.columns = ["time",
				   "flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9",
				   "flow_10",
				   "flow_11",
				   "flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
				   ]
	sub.to_csv(conf.predict_data_path + "linear_sub.csv", index=False)


if __name__ == '__main__':
	data = pd.read_csv(conf.tmp_data_paht+"all_data_new.csv")
	model_test(data)