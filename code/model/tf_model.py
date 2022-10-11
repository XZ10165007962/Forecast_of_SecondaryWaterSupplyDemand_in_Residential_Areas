#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: tf_model.py
@time: 2022/10/9 14:57
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt    #画图的

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers

from code.util import conf
from code.util.util import MSLE

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

data = pd.read_csv(conf.tmp_data_paht+"all_data_new.csv")
feat_col = []
label_col = []
for i in range(24*7, 24*14):
	data[f"flow_lag_{i}"] = data.groupby(["flow_id"])["flow"].shift(i)
	print(f"flow_lag_{i}")
for i in range(24*7):
	data[f"label_{i}"] = data.groupby(["flow_id"])["flow"].shift(-i)
	label_col.append(f"label_{i}")
pre_list = []
test_list = []
pre_df = pd.DataFrame()
for i, time in enumerate(['2022-05-08 00:00:00', '2022-06-08 00:00:00', '2022-07-28 00:00:00', '2022-08-28 00:00:00']):
		for id in [
			"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
			"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
		]:
			print(f"========={id}=============")
			data_ = data[data["time"] <= time]
			use_data = data_[data_["flow_id"] == id]
			use_data.dropna(subset=feat_col, inplace=True)
			train = use_data.iloc[:-24*7, :]
			test = use_data.iloc[-24*7:, :]

			X_train, y_train = train[feat_col], train[label_col]
			X_test, y_test = test[feat_col], test[label_col]

			scaler_X = StandardScaler().fit(X_train)
			scaler_y = StandardScaler().fit(y_train)
			X_train, y_train = scaler_X.transform(X_train), scaler_y.transform(y_train)
			X_test, y_test = scaler_X.transform(X_test), scaler_y.transform(y_test)
			# 可着train集合里面的祸害，虽然无所谓，但是就这么干了
			samples = X_train.shape[0]  # 就是咱们送进去多少个样本
			timesteps = X_train.shape[1]  # 就是咱们的时间步长，就lags是多少
			n_features = 1  # 特徵几个，咱们这里就一种负荷么不，自己予测自己，也就是1

			outputs = y_train.shape[-1]  # 输出多少天的

			X_train = X_train.reshape((samples, timesteps, n_features))  # reshape了，是为了后面的输入用的
			y_train = y_train  # 标签没什么好改的，这里是为了清晰

			X_test = X_test.reshape((X_test.shape[0], timesteps, n_features))  # 这里不行！那个sample得按照X_test的来
			y_test = y_test

			samples, timesteps, n_features = X_train.shape[0], X_train.shape[1], 1
			outputs = y_train.shape[1]
			n_features = 1
			n_classes = y_train.shape[1]

			def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
				# Attention and Normalization
				x = layers.MultiHeadAttention(
					key_dim=head_size, num_heads=num_heads, dropout=dropout
				)(inputs, inputs)
				x = layers.Dropout(dropout)(x)
				x = layers.LayerNormalization(epsilon=1e-6)(x)
				res = x + inputs

				# Feed Forward Part
				x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(res)
				x = layers.Dropout(dropout)(x)
				x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="tanh")(x)
				x = layers.LayerNormalization(epsilon=1e-6)(x)
				return x + res

			def build_model(
					input_shape,
					head_size,
					num_heads,
					ff_dim,
					num_transformer_blocks,
					mlp_units,
					dropout=0,
					mlp_dropout=0,
			):
				inputs = keras.Input(shape=input_shape)
				x = inputs
				for _ in range(num_transformer_blocks):
					x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

				x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
				for dim in mlp_units:
					x = layers.Dense(dim, activation="tanh")(x)
					x = layers.Dropout(mlp_dropout)(x)
				# outputs = layers.Dense(n_classes, activation="softmax")(x)
				outputs = layers.Dense(n_classes)(x)
				return keras.Model(inputs, outputs)


			input_shape = X_train.shape[1:]

			TRS = build_model(
				input_shape,
				head_size=64,
				num_heads=2,
				ff_dim=2,
				num_transformer_blocks=2,
				mlp_units=[258],
				mlp_dropout=0.1,
				dropout=0.1,
			)
			opt = Adamax(epsilon=0.001)
			TRS.compile(loss="mse", optimizer=opt, )
			# TRS.summary()

			epsilon = 0.01
			EPOCHS = 1
			BATCH_SIZE = 64
			callback = EarlyStopping(monitor="val_loss", patience=EPOCHS * .05, restore_best_weights=True)

			TRS.fit(X_train, y_train,
					epochs=EPOCHS, batch_size=BATCH_SIZE,
					verbose=1, validation_split=0.25,
					callbacks=callback)
			pre = TRS.predict(X_test)
			# test_list.extend(scaler_y.inverse_transform(y_test))
			# pre_list.extend(scaler_y.inverse_transform(pre))
			print(pd.DataFrame(scaler_y.inverse_transform(pre)).shape)
			print("mse", mean_squared_error(scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(pre)))

