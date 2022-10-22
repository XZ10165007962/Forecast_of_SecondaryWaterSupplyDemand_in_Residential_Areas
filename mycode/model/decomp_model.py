#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: decomp_model.py
@time: 2022/10/21 11:19
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings

from matplotlib import pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from datetime import timedelta
from sklearn.metrics import mean_absolute_error

from mycode.util import conf
from mycode.util.util import MSLE

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


class ModelDecomp():
	def __init__(self, file, id, test_size=24*7):
		self.id = id
		self.ts = self.read_data(file)
		self.test_size = test_size
		self.train_size = len(self.ts) - self.test_size
		self.train = self.ts[:len(self.ts) - test_size]
		self.test = self.ts[-test_size:]

	def read_data(self, f):
		data = pd.read_csv(f)
		data = data[(data['time'] >= '2022-02-01 01:00:00') & (data['time'] < '2022-05-01 01:00:00')].reset_index(
			drop=True)
		data = data[data["flow_id"] == self.id]
		data = data.set_index('time')
		data.index = pd.to_datetime(data.index)
		ts = data['flow']
		return ts

	def decomp(self, freq):
		"""
		对时间序列进行分解
		:param freq: 周期
		:return:
		"""
		decomposition = seasonal_decompose(self.train, freq=freq, two_sided=False)
		self.trend = decomposition.trend
		self.seasonal = decomposition.seasonal
		self.residual = decomposition.resid
		# decomposition.plot()
		# plt.show()

		d = self.residual.describe()
		delta = d['75%'] - d['25%']
		self.low_error, self.high_error = (d['25%'] - 1 * delta, d['75%'] + 1 * delta)
		self.mean_error = d["50%"]

	def trend_model(self, order):
		"""
		为分解出来的趋势数据单独建模
		"""
		self.trend.dropna(inplace=True)
		self.trend_model = ARIMA(self.trend, order).fit(disp=-1, method='css')

		return self.trend_model

	def add_season(self):
		"""
		为预测数的趋势数据添加周期数据和残差数据
		:return:
		"""
		self.train_season = self.seasonal
		values = []
		low_conf_values = []
		high_conf_values = []

		for i,t in enumerate(self.pred_time_index):
			trend_part = self.trend_pred[i]

			# 相同时间的数据均值
			season_part = self.train_season[
				self.train_season.index.time == t.time()
			].mean()
			# 趋势+周期+误差界限
			predict = trend_part + season_part
			low_bound = trend_part + season_part + self.low_error
			high_bound = trend_part + season_part + self.high_error

			values.append(predict)
			low_conf_values.append(low_bound)
			high_conf_values.append(high_bound)

		self.final_pred = pd.Series(values, index=self.pred_time_index, name='predict')
		self.low_conf = pd.Series(low_conf_values, index=self.pred_time_index, name='low_conf')
		self.high_conf = pd.Series(high_conf_values, index=self.pred_time_index, name='high_conf')

	def predict_new(self):
		"""
		预测新数据
		"""
		# 续接train，生成长度为n的时间索引，赋给预测序列
		n = self.test_size
		self.pred_time_index = pd.date_range(start=self.train.index[-1], periods=n + 1, freq='1H')[1:]
		self.trend_pred = self.trend_model.forecast(n)[0]
		self.add_season()


def evaluate(filename, id, if_show=False):
	md = ModelDecomp(filename, id)
	md.decomp(freq=24)
	md.trend_model(order=(1,1,5))
	md.predict_new()
	pred = md.final_pred
	test = md.test
	if if_show:
		plt.subplot(211)
		plt.plot(md.ts)
		plt.title(filename.split('.')[0])
		plt.subplot(212)
		pred.plot(color='salmon', label='Predict')
		test.plot(color='steelblue', label='Original')
		# md.low_conf.plot(color='grey', label='low')
		# md.high_conf.plot(color='grey', label='high')

		plt.legend(loc='best')
		plt.title('RMSE: %.4f' % np.sqrt(sum((pred.values - test.values) ** 2) / test.size))
		plt.tight_layout()
		plt.show()


if __name__ == '__main__':
	file = conf.tmp_data_paht+"all_data_new.csv"
	# flag = 1
	# if flag:
	# 	test_list = []
	# 	pre_list = []
	# 	for id in [
	# 		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
	# 		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	# 	]:
	# 		model = ModelDecomp(file, id)
	# 		model.decomp(freq=24)
	# 		model.trend_model(order=(1, 1, 5))
	# 		model.predict_new()
	# 		test_list.extend(model.test)
	# 		pre_list.extend(model.final_pred)
	# 		print("mae:",mean_absolute_error(model.test, model.final_pred))
	# 		print(MSLE(test_list, pre_list, flag=1))
	# else:
	# 	evaluate(file, "flow_4", if_show=True)
	data = pd.read_csv(file)
	data["time"] = pd.to_datetime(data["time"])
	data["day"] = data["time"].dt.dayofyear
	data["flow"] = data.groupby(["flow_id", "day"])["flow"].transform("mean")
	data = data.drop_duplicates()
	print(data.head())