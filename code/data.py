# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         data
# Description:
# Author:       xinzhuang
# Date:         2022/9/12
# Function:
# Version：
# Notice:
# -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import conf

# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)

"""
整个数据 2022-01-01 01:00:00 开始
训练集划分为
test1 开始 2022-04-01 01:00:00 2161  结束 2022-04-08 00:00:00  2328
"""
"""
test1 开始 2022-05-01 01:00:00 2885  结束 2022-05-08 00:00:00  3048
test2 开始 2022-06-01 01:00:00 3629  结束 2022-06-08 00:00:00  3792
test3 开始 2022-07-21 01:00:00 4829  结束 2022-07-28 00:00:00  4992
test4 开始 2022-08-21 01:00:00 5573  结束 2022-08-28 00:00:00  5736
"""


def get_data():
	data = pd.read_csv(conf.train_data_path + "hourly_dataset.csv")
	data["time_index"] = np.arange(1, data.shape[0] + 1)
	data = data[data["time_index"] <= 2328]
	return data


def split_data(data_, split_col, split_flag, label_col):
	train_data = data_[data_[split_col] < split_flag]
	test_data = data_[data_[split_col] >= split_flag]
	feature_col = [i for i in data_.columns if i != label_col]
	train_x = train_data.loc[:, feature_col]
	train_y = train_data[label_col]
	test_x = test_data.loc[:, feature_col]
	test_y = test_data[label_col]

	return train_x, train_y, test_x, test_y