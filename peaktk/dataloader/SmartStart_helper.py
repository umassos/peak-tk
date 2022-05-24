
import glob
import math
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from peaktk.preprocessing import helper_isone
from peaktk.preprocessing import preprocessing

def load_data(DEMAND_PATH, WEATHER_PATH, years=[2015, 2016]):
	# load Smart* dataset
	path = DEMAND_PATH
	all_df = []
	for y in years:
	    y_path = path + str(y)
	    all_files = sorted(glob.glob(y_path + "/*.csv"))
	    for filename in all_files:
	        all_df.append(pd.read_csv(filename, header=None, names=["datetime","load"], parse_dates = ["datetime"], index_col=0))
	smartstar_df = pd.concat(all_df).dropna()

	# isone_df.index = isone_df.index.tz_localize('US/Eastern', ambiguous=infer_dst)

	infer_dst = np.array([False] * smartstar_df.shape[0])
	smartstar_df.index = smartstar_df.index.tz_localize('US/Eastern', ambiguous=infer_dst)

	smartstart_daily_df = smartstar_df.groupby(smartstar_df.index.date).max()

	# load smartstar weather data
	path = WEATHER_PATH
	all_df = []
	all_files = glob.glob(path + "/*.csv")
	for filename in all_files:
	    temp_df = pd.read_csv(filename, index_col="time", parse_dates=["time"])
	    infer_dst = np.array([False] * temp_df.shape[0])
	    temp_df.index = pd.to_datetime(temp_df.index, unit='s').tz_localize('UTC', ambiguous=infer_dst).tz_convert('US/Eastern')
	    all_df.append(temp_df)
	smartstar_weather_df = pd.concat(all_df)

	smartstar_weather_df = smartstar_weather_df.groupby(smartstar_weather_df.index.date).agg({
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
	smartstar_weather_df.columns = [
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
	smartstar_weather_df.index.name = "date"

	# merge
	smartstar_merge_df = pd.concat([smartstart_daily_df, smartstar_weather_df], axis=1, join="inner")
	smartstar_merge_df.index = pd.to_datetime(smartstar_merge_df.index)


	median = smartstar_merge_df["load"].median()
	std = smartstar_merge_df["load"].std()
	outliers = (smartstar_merge_df["load"] - median).abs() > std*1.2
	smartstar_merge_df[outliers] = median

	smartstar_merge_df.rename({"load":"Load"}, axis=1, inplace=True)

    smartstar_merge_df.index.name = "date"

    return smartstar_merge_df







