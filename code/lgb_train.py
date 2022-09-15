#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: lgb_train.py
@time: 2022/9/15 11:15
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import requests

import data
import features

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


if __name__ == '__main__':
    data = data.get_data()
    data, feature = features.get_features(data)