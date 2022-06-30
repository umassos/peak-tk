
import math
import pandas as pd
import numpy as np
import datetime
from calendar import monthrange
import scipy.stats
from functools import reduce
import operator
from sklearn.metrics import confusion_matrix


class ProbabilisticApproach:
    def __init__(self):
        pass

    def fit(self, historical_df, current_year, N=5, look_ahead=3, quantile=0.83, extreme_temp=(41,86), threshold_p=0.1, delta=0, lock_threshold=False):

        self.peak_day_picked = 0
        # self.historical_df = daily_df[(daily_df.index.year<current_year)]
        self.historical_df = historical_df
        self.historical_df["dayofyear"] = self.historical_df.index.strftime("%j").astype('int64').tolist()
        
        self.threshold = self.historical_df["Load"].quantile(quantile)
        self.hist_peak_load = []
        # MODIFY TO FIX ONTARIO COLD START
        self.hist_peak_load = self.historical_df.nlargest(N*2, 'Load')['Load'].values[:N]
        self.look_ahead = look_ahead
        self.threshold_p = threshold_p
        self.N = N
        self.delta = delta

        return self

    def predict(self, predicted_demand, tomorrow_temperature, current_date):
        is_pred_peak = False

        # all peak day already predicted
        if self.peak_day_picked >= (self.N+self.delta):
            return is_pred_peak

        tomorrow_load = predicted_demand[0]
        is_extreme_weather = self.check_extreme_temp(tomorrow_temperature, extreme_temp)

        if daily_load > self.threshold or is_extreme_weather:
            # FIND STD, PRED_DEMAND
            std = []
            pred_demand = []
            idx = current_date.timetuple().tm_yday
            std.append(self.historical_df[self.historical_df.index.dayofyear == idx+1]["Load"].std())
            pred_demand.append(tomorrow_load)
            for j in range(1, self.look_ahead):
                # prevent index out of bound
                if j < len(predicted_demand):
                    future_dt = current_date + datetime.timedelta(day=j)
                    # print(future_dt, future_dt.day, future_dt.month)
                    # std.append(historical_df[(historical_df.index.day == future_dt.day) & (historical_df.index.month == future_dt.month)]["MaxPower"].std())
                    std.append(self.historical_df[(self.historical_df.index.dayofyear == idx+1+j)]["Load"].std())
                    pred_demand.append(predicted_demand[j])
            std = np.cumsum(std)
            # print(N, look_ahead, pred_demand, std, hist_peak_load, threshold_p)
            # PREDICT
            is_pred_peak = self.compute_prob(self.N, self.look_ahead, pred_demand, std, self.hist_peak_load, self.threshold_p)
            ## APPEND IS PEAK
            # pred_peak.append(is_pred_peak)
            ## CHECK IF CAN TERMINATE
            if is_pred_peak:
                self.peak_day_picked += 1
        elif not lock_threshold and daily_load > threshold and not is_extreme_weather:
            self.threshold = daily_load
        elif not lock_threshold and daily_load < threshold and is_extreme_weather:
            self.threshold = daily_load

        return is_pred_peak

    def probabilistic_model(self, y_pred, temp_pred, daily_df, current_year, N=5, look_ahead=3, quantile=0.83, extreme_temp=(41,86), threshold_p=0.1, delta=0, lock_threshold=False):
        pred_peak = []
        # gt_peak = []
        # recall = []
        # daily_df["y_true"] = y_true

        peak_day_picked = 0
        historical_df = daily_df[(daily_df.index.year<current_year)]
        historical_df["dayofyear"] = historical_df.index.strftime("%j").astype('int64').tolist()
        daily_temperature = temp_pred
        
        threshold = historical_df["Load"].quantile(quantile)
        hist_peak_load = []
        # MODIFY TO FIX ONTARIO COLD START
        hist_peak_load = historical_df.nlargest(N*2, 'Load')['Load'].values[:N]

        for idx, daily_load in enumerate(y_pred):

            # is_extreme_weather = self.check_extreme_weather(daily_temperature[idx], historical_df, weather_quantile)
            is_extreme_weather = self.check_extreme_temp(daily_temperature[idx], extreme_temp)

            # print(idx, is_extreme_weather, daily_load, threshold, (daily_load > threshold), daily_temperature[idx])

            if daily_load > threshold or is_extreme_weather:
                # FIND STD, PRED_DEMAND
                std = []
                pred_demand = []
                std.append(historical_df[historical_df.index.dayofyear == idx+1]["Load"].std())
                pred_demand.append(daily_load)
                for j in range(1, look_ahead):
                    # prevent index out of bound
                    if idx+j < len(y_pred):
                        future_dt = datetime.datetime(current_year, 1, 1) + datetime.timedelta(idx+j)
                        # print(future_dt, future_dt.day, future_dt.month)
                        # std.append(historical_df[(historical_df.index.day == future_dt.day) & (historical_df.index.month == future_dt.month)]["MaxPower"].std())
                        std.append(historical_df[(historical_df.index.dayofyear == idx+1+j)]["Load"].std())
                        pred_demand.append(y_pred[idx+j])
                std = np.cumsum(std)
                # print(N, look_ahead, pred_demand, std, hist_peak_load, threshold_p)
                # PREDICT
                is_pred_peak = self.compute_prob(N, look_ahead, pred_demand, std, hist_peak_load, threshold_p)
                ## APPEND IS PEAK
                pred_peak.append(is_pred_peak)
                ## CHECK IF CAN TERMINATE
                if is_pred_peak:
                    peak_day_picked += 1
                    if peak_day_picked >= (N+delta):
                        pred_peak += [False]*(len(y_pred)-(idx+1))
                        break
                #     hist_peak_load.append(daily_load)
                # hist_peak_load.sort(reverse=True)

    #                 if P(R_overall[K]) >= threshold_p:
    #                     # tomorrow will be peak day
    #                     is_pred_peak = True

                continue
                    
            elif not lock_threshold and daily_load > threshold and not is_extreme_weather:
                threshold = daily_load
            elif not lock_threshold and daily_load < threshold and is_extreme_weather:
                threshold = daily_load

            pred_peak.append(False)

        return pred_peak

    def find_datetime_from_indexofyear(self, indexofyear, current_year):
        return datetime.datetime(current_year, 1, 1) + datetime.timedelta(indexofyear)

    def check_extreme_weather(self, today_temp, year_temp, quant=0.9):
        if today_temp < year_temp["MaxTemp"].quantile(1-quant):
            return True
        if today_temp > year_temp["MaxTemp"].quantile(quant):
            return True
        return False

    def check_extreme_temp(self, today_temp, extreme_temp):
        if today_temp <= extreme_temp[0]:
            return True
        if today_temp >= extreme_temp[1]:
            return True
        return False
        
    # https://stackoverflow.com/questions/12412895/how-to-calculate-probability-in-a-normal-distribution-given-mean-standard-devi
    def compute_prob(self, N, K, pred_demand, std, hist_peak_load, threshold_p):

        # NEED MEAN, AND STD
        # TOMORROW VS FUTURE
    #     pred_demand = [23665,23932, , ]
    #     std = [210, 584]
        p_compare_future = [1]*len(pred_demand)
    #     p_compare_future[0] = 0
    #     a = NormalDist(0,std[0] + std[1])
    #     p_compare_future[1] = a.cdf(pred_demand[0]-pred_demand[1])
        for i in range(1, len(pred_demand)):
            # a = NormalDist(0,std[0] + std[i])
            a = scipy.stats.norm(0,std[0] + std[i])
            p_compare_future[i] = a.cdf(pred_demand[0]-pred_demand[i])
        # print("p_compare_future")
        # print(p_compare_future)
        # PEAK VS TOMORROW
    #     hist_peak_load = [24636, 24107, 23910, 23801, 23745]
        # a = NormalDist(0,std[0])
        a = scipy.stats.norm(0,std[0])
    #     p_compare_peak[0] = 1 - a.cdf(pred_demand[0]-hist_peak_load[0])
    #     p_compare_peak[1] = 1 - a.cdf(pred_demand[0]-hist_peak_load[1])
        p_compare_peak = []
        for i in range(0, min(len(hist_peak_load), N)):
            p =  1 - a.cdf(pred_demand[0]-hist_peak_load[i])
            p_compare_peak.append(p)
        # print("p_compare_peak")
        # print(p_compare_peak)
        # P(RANK FUTURE)
        # P RANK FUTURE == 1, ALL P MULTIPLY
        p_rank_future = [1]*(len(p_compare_future))
        if len(p_compare_future) > 0:
            p_rank_future[0] = self.prod(p_compare_future)
            for i in range(1, len(p_compare_future)-1):
                p_rank_future[i] = (1 - np.sum(p_rank_future[:i]))*self.prod(p_compare_future[i+1:])
            p_rank_future[len(p_compare_future)-1] = (1 - np.sum(p_rank_future[:i]))
            # print("p_rank_future")
            # print(p_rank_future)

        ## P RANK PAST
        p_rank_past = [1/N]*N
        # p_rank_past[0] = 1
        if len(p_compare_peak) > 0:
            p_rank_past[0] = 1 - p_compare_peak[0]
        #     p_rank_past[1] = (1 - p_compare_peak[1]) - p_rank_past[0]
        #     p_rank_past[2] = (1 - p_compare_peak[2]) - p_rank_past[1]
        #     p_rank_past[3] = (1 - p_compare_peak[3]) - p_rank_past[2]
            for i in range(1, len(p_compare_peak)):
                p = (1 - p_compare_peak[i]) - p_rank_past[i-1]
                p_rank_past.append(p)

        # P RANK OVERALL
        p_rank_overall = []
        for i in range(0, N):
            goal = i+1
            prob = 0
            for j in range(0, len(p_rank_past)):
                for k in range(0, len(p_rank_future)):
                    if j+k == goal:
                        prob += p_rank_past[j]*p_rank_future[k]
            p_rank_overall.append(prob)
        # print("p_rank_overall")
        # print(p_rank_overall)
        # P RANK OVER <= K
        prob_rank_k = sum(p_rank_overall)
        # print(prob_rank_k)
        # exit()
        if prob_rank_k > threshold_p:
            return True
        return False

    def normpdf(self, x, mean, sd):
        # https://stackoverflow.com/a/12413491
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom

    def prod(self, iterable):
        return reduce(operator.mul, iterable, 1) 

    # p, r, a = probabilistic_model(y_pred_month, y_true, daily_df, N=5, look_ahead=3, quantile=0.9, threshold_p=0.1, delta=0)
    