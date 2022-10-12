#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: time_model.py
@time: 2022/10/9 20:58
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from mycode.util import conf, util
import pmdarima as pm

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

data = pd.read_csv(conf.tmp_data_paht+"all_data_new.csv")
train1 = data[(data['time']>='2022-01-01 01:00:00')&(data['time']<'2022-05-01 01:00:00')].reset_index(drop=True)
pre_list = []
for id in [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]:
	train_data = train1[(train1['time']>='2022-02-15 01:00:00')&(train1['time']<'2022-04-24 01:00:00')]
	test_data = train1[(train1['time']>='2022-04-24 01:00:00')&(train1['time']<'2022-04-25 01:00:00')]
	train_data = train_data[train_data["flow_id"] == id]
	test_data = test_data[test_data["flow_id"] == id]
	model = ExponentialSmoothing(
		train_data["flow"].values,
		seasonal_periods=24,
		trend="add",
		seasonal="add",
	).fit()
	pre = model.forecast(len(test_data))
	pre_list.extend(pre)
	print(mean_squared_error(pre, test_data["flow"].values))
print("msle",util.MSLE(train1[(train1['time']>='2022-04-24 01:00:00')&(train1['time']<'2022-04-25 01:00:00')]["flow"].values.tolist(), pre_list, flag=1))