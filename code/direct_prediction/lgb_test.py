#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: lgb_test.py
@time: 2022/9/22 14:14
@version:
@desc: 
"""
import numpy as np
import pandas as pd
import warnings

import conf
import features
import tree_model
from code.direct_prediction import data

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
    test_list = ["test1", "test2", "test3", "test4"]
    for test_num in test_list:
        test = all_data[all_data["train or test"] == test_num]
        time_min = test["time_index"].min()
        val_time = time_min - 24 * 7
        train_time = val_time - 24 * 7
        val_data = all_data[(all_data["time_index"] < time_min) & (all_data["time_index"] >= val_time)]
        train_data = all_data[(all_data["time_index"] < val_time) & (all_data["time_index"] >= train_time)]
        val_x = val_data.loc[:, feature]
        val_y = val_data.loc[:, label]
        train_x = train_data.loc[:, feature]
        train_y = train_data.loc[:, label]
        test_x = test.loc[:, feature]
        val_pred, test_pred = tree_model.lgb_model(train_x, train_y, test_x, val_x, val_y, cat)
        all_data.loc[test.index, ["flow"]] = test_pred
    all_data.to_csv(conf.predict_data_path + "sub_all.csv", index_label=False)
    all_data = all_data[all_data["train or test"] != "train"]
    flow_id = [
        "flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
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
                   "flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10",
                   "flow_11",
                   "flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
                   ]
    sub.to_csv(conf.predict_data_path + "sub.csv", index=False)