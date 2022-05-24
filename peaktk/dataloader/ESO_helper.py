
import glob
import math
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from peaktk.preprocessing import helper_isone
from peaktk.preprocessing import preprocessing

def load_data(DEMAND_PATH, WEATHER_PATH):
	# load ESO dataset
	path = DEMAND_PATH
	all_df = []
	all_files = sorted(glob.glob(path + "/*.csv"))
	for filename in all_files:
	    print(filename)
	    all_df.append(pd.read_csv(filename, header=0, parse_dates = ['SETTLEMENT_DATE'], index_col=0))
	eso_df = pd.concat(all_df).dropna()

	# convert to daily hourly, 
	eso_daily_df = eso_df.groupby(by=eso_df.index.date).max()[["ENGLAND_WALES_DEMAND"]]

	# load ESO weather data
	path = WEATHER_PATH
	all_df = []
	all_files = glob.glob(path + "/*.csv")
	for filename in all_files:
	    temp_df = pd.read_csv(filename, index_col="time", parse_dates=["time"])
	    infer_dst = np.array([False] * temp_df.shape[0])
	    temp_df.index = pd.to_datetime(temp_df.index, unit='s').tz_localize('UTC', ambiguous=infer_dst).tz_convert('US/Eastern')
	    all_df.append(temp_df)
	eso_weather_df = pd.concat(all_df)

	eso_weather_daily_df = eso_weather_df.groupby(eso_weather_df.index.date).agg({
	        'temperature':['max','min']
	        ,'humidity':['max','min']
	        ,'apparentTemperature':['max','min']
	        ,'pressure':['max','min']
	        ,'windSpeed':['max','min']
	        ,'cloudCover':['max','min']
	        ,'windBearing':['max','min']
	        ,'dewPoint':['max','min']
	        ,'precipIntensity':['max','min']
	        ,'precipProbability':['max','min']})
	eso_weather_daily_df.columns = [
	        "Max_temperature","Min_temperature"
	        ,"Max_humidity","Min_humidity"
	        ,"Max_apparentTemperature","Min_apparentTemperature"
	        ,"Max_pressure","Min_pressure"
	        ,"Max_windSpeed","Min_windSpeed"
	        ,"Max_cloudCover","Min_cloudCover"
	        ,"Max_windBearing","Min_windBearing"
	        ,"Max_dewPoint","Min_dewPoint"
	        ,"Max_precipIntensity","Min_precipIntensity"
	        ,"Max_precipProbability","Min_precipProbability"
	        ]
	eso_weather_daily_df.index.name = "date"

	eso_weather_daily_df.drop(['Max_pressure', 'Min_pressure', 'Max_precipIntensity', 'Min_precipIntensity', 'Max_precipProbability', 'Min_precipProbability'], axis=1, inplace=True)
	eso_weather_daily_df.fillna(method="ffill", inplace=True)

	# MERGE
	eso_merge_df = pd.concat([eso_weather_daily_df,                   
	                          eso_daily_df], axis=1, join="inner")
	eso_merge_df.index = pd.to_datetime(eso_merge_df.index)

	eso_merge_df.rename({"ENGLAND_WALES_DEMAND":"Load"}, axis=1, inplace=True)

	eso_merge_df.index.name = "date"

	return eso_merge_df

