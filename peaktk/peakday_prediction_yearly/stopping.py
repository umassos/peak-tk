

class StoppingApproach:
	def __init__(self, m, k, threshold_k=999, limit_peakday=True, stingy=False):
		self._m = m
		self._k = k
		self._counter = 0
		self._picked_peak = 0
		self._max = 0
		self._topK = []
		self._current_year = -1
		self._limit_peakday = limit_peakday
		self._threshold_k = min(threshold_k, k, m)
		# if threshold_k > k:
		# 	self._threshold_k = k
		self._stingy = stingy

	def fit(self):
		# do nothing
		return self

	def predict(self,tomorrow_predicted_demand, current_date):
		is_peak = False
		if self._current_year != current_date.year:
			self.reset_params(current_date.year)


		if self._counter <= self._m:
			# learning phase
			if len(self._topK) < self._k:
				self._topK.append(tomorrow_predicted_demand)
				self._topK.sort(reverse=True)
			elif tomorrow_predicted_demand > self._topK[-1]:
				# TODO: performance can be improved by using a better data structure
				# replace lowest and sort
				self._topK[-1] = tomorrow_predicted_demand
				self._topK.sort(reverse=True)
		else:
			# picking phrase
			if tomorrow_predicted_demand > self._topK[self._threshold_k-1]:
				if self._limit_peakday and self._picked_peak >= self._k:
					# limit number of peak day that can be picked to k
					is_peak = False
				else:
					if self._stingy:
						# not pick anything lower than prev peak
						self._topK[self._threshold_k-1] = tomorrow_predicted_demand
					is_peak = True

				self._picked_peak += 1
			else:
				is_peak = False
		self._counter += 1

		return is_peak

	def reset_params(self, year):
		# new month, reset some params
		self._counter = 0
		self._max = 0
		self._current_year = year
		self._topK = []

