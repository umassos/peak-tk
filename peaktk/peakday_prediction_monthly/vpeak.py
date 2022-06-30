# Author: Nikko 2021
import pandas as pd
import numpy as np
import h5py

class VPeak_Selection:
	def __init__(self):
		self.monthly_threshold = [] # 1d list
		self.monthly_cdf = [] # 2d list
		self.current_month = -1
		self.current_threshold = 0
		self.top_k = 0
		self.picked = 0
		self.update_frequency = 1 # every 1 day
		self.alpha = 0
		self.beta = 0
		self.stingy = False
		self.min_peak = -1
		self.dynamic_thres = False
		pass

	def train(self, historical_daily_demand, one_year_daily_demand, top_k, quantile=0.86, update_frequency=5, alpha=0.01, beta=0.01, dynamic_thres=False, stingy=False):
		#input: monthly_historical_demand (12, x)
		if len(historical_daily_demand) != 12:
			print("ERROR: expected data of shape (12, x), 12 months")
			return

		for i in range(0,12):
			# find threshold
			h_demand = np.array(historical_daily_demand[i])
			threshold = np.quantile(h_demand,quantile)
			self.monthly_threshold.append(threshold)

			d_demand = np.array(one_year_daily_demand[i])
			# calculate cdf 
			num_of_day = len(d_demand)
			hist, bins = np.histogram(d_demand.argsort()[::-1][:top_k],bins=np.linspace(0,num_of_day,num_of_day+1))
			cdf = hist.cumsum()/top_k
			self.monthly_cdf.append(cdf.tolist())

		self.top_k = top_k
		self.alpha = alpha
		self.beta = beta
		self.stingy = stingy
		self.update_frequency = update_frequency
		self.dynamic_thres = dynamic_thres

	def reset_params(self, new_current_month, new_current_threshold):
		self.picked = 0
		self.min_peak = -1
		self.current_month = new_current_month
		# print(self.current_month)
		self.current_threshold = new_current_threshold


	def predict_ispeak(self, current_date, demand):

		# monthly reset 
		if current_date.month != self.current_month:
			self.reset_params(current_date.month, self.monthly_threshold[self.current_month-1])

		# all peak day is picked
		if self.picked >= self.top_k:
			return False

		# stingy mode, not picking anything lower than previous peak
		if self.stingy and demand < self.min_peak:
			return False	
		curr_day = current_date.day-1
		curr_month = self.current_month-1
		# should we update threshold? (every x days)
		if self.dynamic_thres and (current_date.day % self.update_frequency == 0):
			# TODO: Feb can have 29 days
			cdf = self.monthly_cdf[curr_month]
			if self.picked/self.top_k > cdf[curr_day]:
				# pick too much, increase the threshold
				self.current_threshold *= (1+self.alpha) 
			elif self.picked/self.top_k < cdf[curr_day]:
				# pick too less, decrease the threshold
				self.current_threshold *= (1-self.beta) 

		if demand > self.current_threshold:
			self.picked += 1
			self.min_peak = demand
			# less than 85% of the demand, threshold is definitely too low
			if self.current_threshold < demand*0.85:
				self.current_threshold = demand*0.85
			return True

		return False

	def save_model(self, filename):
		#Writing data
		try:
			hf = h5py.File(filename, 'w')
			for i in range(0, 12):
				dset1 = hf.create_dataset(str(i), data=self.monthly_cdf[i])

			hf.create_dataset("monthly_threshold", data=self.monthly_threshold)

			data_dict={	"top_k":self.top_k, 
						"alpha":self.alpha, 
						"beta":self.beta, 
						"stingy":self.stingy, 
						"update_frequency":self.update_frequency, 
						"dynamic_thres":self.dynamic_thres,
						"picked":self.picked,
						"min_peak":self.min_peak,
						"current_month":self.current_month,
						"current_threshold":self.current_threshold,
						}

			#Store this dictionary object as hdf5 metadata
			for k in data_dict.keys():
				hf.attrs[k]=data_dict[k]
		finally:
			hf.close()

	def load_model(self, filename):
		#Reading data
		try:
			hf = h5py.File(filename, 'r')
			# print(hf.attrs.keys())

			data = []
			for i in range(0, 12):
				data.append(list(hf.get(str(i))))
			self.monthly_cdf = data
			# print(self.monthly_cdf)
			self.monthly_threshold = list(hf.get("monthly_threshold"))
			# print(self.monthly_threshold)

			# parameters
			self.top_k = hf.attrs["top_k"]
			self.update_frequency = hf.attrs["update_frequency"]
			self.alpha = hf.attrs["alpha"]
			self.beta = hf.attrs["beta"]
			self.stingy = hf.attrs["stingy"]
			self.dynamic_thres = hf.attrs["dynamic_thres"]
		finally:
			hf.close()

	def save_state(self, filename):

		try:
			hf = h5py.File(filename, 'w')
			# d1 = np.random.random(size = (1,1))  #sample data
			# hf.create_dataset("monthly_threshold", data=self.monthly_threshold)

			data_dict={	"picked":self.picked,
						"min_peak":self.min_peak,
						"current_month":self.current_month,
						"current_threshold":self.current_threshold,
						}

			#Store this dictionary object as hdf5 metadata
			for k in data_dict.keys():
				hf.attrs[k]=data_dict[k]
		finally:
			hf.close()
		

	def load_state(self, filename):
		#Reading data
		try:
			hf = h5py.File(filename, 'r')

			# states
			self.picked = hf.attrs["picked"]
			self.min_peak = hf.attrs["min_peak"]
			self.current_month = hf.attrs["current_month"]
			self.current_threshold = hf.attrs["current_threshold"]

		finally:
			hf.close()

	def reset_state(self):
		self.picked = 0
		self.min_peak = -1
		self.current_month = -1
		self.current_threshold = 0

