#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: tree_train.py
@time: 2022/9/15 11:15
@version:
@desc: 
"""
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

from code.direct_prediction import data
import features
import tree_model
import conf
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
    all_data = pd.read_csv(conf.tmp_data_paht + "all_data.csv")
    all_data, feature, cat = features.get_features(all_data)
    label = ["flow"]
    time_start = 2713
    time_end = 2880
    data = all_data[all_data["time_index"] <= time_end]
    test_data = data[data["time_index"] >= time_start]
    val_start = time_start - 24*7
    val_end = time_end - 24*7
    val_data = data[(data["time_index"] >= val_start) & (data["time_index"] >= val_end)]
    train_data = data[data["time_index"] < val_start]
    train_x = train_data.loc[:, feature]
    train_y = train_data.loc[:, label]
    val_x = val_data.loc[:, feature]
    val_y = val_data.loc[:, label]
    test_x = test_data.loc[:, feature]
    test_y = test_data.loc[:, label]
    val_pred, test_pred = tree_model.lgb_model(train_x, train_y, test_x, val_x, val_y, cat)
    all_data.loc[test_x.index, ["pre"]] = test_pred
    print("测试集mae:", mean_absolute_error(test_y.values, test_pred))
    print("测试集mse:", mean_squared_error(test_y.values, test_pred))
    print("测试集msle:", MSLE(test_y.values, test_pred))
    all_data.to_csv(conf.tmp_data_paht + "pre_data.csv", index=False)