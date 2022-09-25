#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: lgb_test.py
@time: 2022/9/22 14:14
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

import conf
import features
import tree_model
import data
from util import MSLE

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

if __name__ == '__main__':
    print("获取数据")
    all_data_ = data.all_data()
    label = ["label"]
    time_index_list = [2880, 3620, 4820, 5560]
    for time_index_ in time_index_list:
        for i in range(24*7+1):
            print(f"=================={i}=====================")
            time_index = time_index_ + i
            all_data = data.get_data(all_data_, time_index)
            all_data, feature = features.get_features(all_data)
            # if i == 0:
            #     all_data.to_csv(conf.tmp_data_paht+"feature_data.csv", index=False)
            train_x, train_y, val_x, val_y, test_x, test_y = data.split_data(all_data, "time_index", time_index, "label", feature)
            val_pred, test_pred = tree_model.lgb_model(train_x, train_y, test_x, val_x, val_y)
            # 将预测数据拼接回原始数据
            print(test_pred)
            test_index = all_data_[all_data_["time_index"] == time_index]
            test_index = test_index.index + 1
            all_data_.loc[test_index, ["flow"]] = test_pred
            if (i != 0) and (i % 7 == 0):
                all_data.to_csv(conf.tmp_data_paht + f"test_data_{i}.csv", index=False)
    all_data_.to_csv(conf.tmp_data_paht + "test_data.csv", index=False)