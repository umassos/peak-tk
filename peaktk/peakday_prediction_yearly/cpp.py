
import pandas as pd
import numpy as np
import operator
import functools

class CPPApproach:
	def __init__(self, num_peakday_per_year=3, quantile=0.991, delta=1000, theta=None, update_day=[1,15], skipped_month=[]):
		self.P = 0 # how many peak days called up to now
		self._threshold = 0 # initial value of τD
		self._current_threshold = 0
		self._delta = 0 # threshold adjustment δ
		self._theta = None
		self._H = [] # average number of actual peak days that have occurred by day in past years
		self._current_year = 0

		self._delta = delta
		self._theta = theta
		if self._theta is None:
			self._theta = delta
		self._update_day = update_day
		self._num_peakday_per_year = num_peakday_per_year
		self._quantile = quantile
		self._skipped_month = skipped_month # month to skip, not predict peak and update threshold at all

	def fit(self, previous_years_demand):
		# theta for increasing the threshold
		# delta for decreasing the threshold

		# find threshold
		flatten_demand = functools.reduce(operator.iconcat, previous_years_demand, [])
		threshold = np.quantile(flatten_demand, self._quantile)
		# expected_rank = int(num_peakday_per_year/(len(flatten_demand)/365))
		# threshold = np.partition(flatten_demand, -expected_rank)[-expected_rank]
		self._threshold = threshold

		h_i_list = []
		for prev_y_demand in previous_years_demand:
			# find top n peakest days
			ind = np.argpartition(prev_y_demand, -self._num_peakday_per_year)[-self._num_peakday_per_year:]
			num_of_day = len(prev_y_demand)
			hist, bins = np.histogram(ind, bins=np.linspace(0,num_of_day,num_of_day+1))
			h_i = hist.cumsum()
			# add one more day, in case of, 366
			h_i = np.append(h_i[:365],(h_i[-1]))
			# h_i_list.append(h_i[:365].tolist())
			h_i_list.append(h_i)
		# print(h_i_list)
		self._H = np.around(np.mean(h_i_list, axis=0))


	def predict(self, tomorrow_demand, current_date):
		if self._current_year != current_date.year:
			self.reset_params(current_date.year)

		if self.P >= self._num_peakday_per_year:
			return False

		if current_date.month in self._skipped_month:
			return False

		if tomorrow_demand > self._current_threshold:
			# predict tomorrow will be peak day
			self.P += 1
			return True
		elif current_date.day in self._update_day:
			H_i = self._H[current_date.timetuple().tm_yday-1]
			if self.P > H_i:
				self._current_threshold += self._theta
			elif self.P < H_i:
				self._current_threshold -= self._delta
		return False

	def reset_params(self, year):
		# new month, reset some params
		self._current_year = year
		self._current_threshold = self._threshold
		self.P = 0

