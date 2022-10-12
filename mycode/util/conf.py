#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: conf.py
@time: 2022/10/8 14:21
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

train_data_path = "../../data/use_data/training_dataset/"
test_data_path = "../../data/use_data/"
predict_data_path = "../../prediction_result/"
tmp_data_paht = "../../data/tmp_data/"