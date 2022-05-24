
import math
import pandas as pd
import numpy as np
from calendar import monthrange
import scipy.stats
from functools import reduce
import operator
from sklearn.metrics import confusion_matrix


class Probabilistic_Approach:
    def __init__(self):
        pass

    def probabilistic_model(self, y_pred_month, daily_df, current_year, N=5, look_ahead=3, quantile=0.83, threshold_p=0.1, delta=0):
        pred_peak = []
        # gt_peak = []
        # recall = []
        # daily_df["y_true"] = y_true
        counter = 0
        for month in range(1, 13):

            peak_day_picked = 0
            monthly_df = daily_df[(daily_df.index.month==month) & (daily_df.index.year==current_year)].copy()
            daily_temperature = monthly_df["MaxTemp"].values
            historical_month_df = daily_df[(daily_df.index.month==month) & (daily_df.index.year<current_year)]
            
            threshold = monthly_df["MaxPower"].quantile(quantile)
            hist_peak_load = []
            # MODIFY TO FIX ONTARIO COLD START
            hist_peak_load = historical_month_df.nlargest(N*2, 'MaxPower')['MaxPower'].values[:N]

            num_day_in_month = monthrange(current_year, month)[1]
            # monthly_df['y_true'] = y_true[counter:counter+num_day_in_month]
            # monthly_df['gt_peak'] = False
            # monthly_df.loc[monthly_df["y_true"] >= monthly_df.nlargest(N, 'y_true')["y_true"].values[-1], "gt_peak"] = True
            # gt_peak += monthly_df["gt_peak"].values.tolist()
            counter += num_day_in_month

            for idx, daily_load in enumerate(y_pred_month[month]):

                is_extreme_weather = self.check_extreme_weather(daily_temperature[idx], historical_month_df, quantile)

                if daily_load > threshold or is_extreme_weather:
                    # FIND STD, PRED_DEMAND
                    std = []
                    pred_demand = []
                    std.append(historical_month_df[historical_month_df.index.day == idx+1]["MaxPower"].std())
                    pred_demand.append(daily_load)
                    for j in range(1, look_ahead):
                        if idx+j < len(y_pred_month[month]):
                            std.append(historical_month_df[historical_month_df.index.day == idx+1+j]["MaxPower"].std())
                            pred_demand.append(y_pred_month[month][idx+j])
                    std = np.cumsum(std)
                    # PREDICT
                    is_pred_peak = self.compute_prob(N, look_ahead, pred_demand, std, hist_peak_load, threshold_p)
                    pred_peak.append(is_pred_peak)
                    ## APPEND NEW PEAK DAY
                    if is_pred_peak:
                        peak_day_picked += 1
                        if peak_day_picked >= (N+delta):
                            pred_peak += [False]*(num_day_in_month-(idx+1))
                            break
                    #     hist_peak_load.append(daily_load)
                    # hist_peak_load.sort(reverse=True)

    #                 if P(R_overall[K]) >= threshold_p:
    #                     # tomorrow will be peak day
    #                     is_pred_peak = True

                    continue
                        
                elif daily_load > threshold and not is_extreme_weather:
                    threshold = daily_load
                elif daily_load < threshold and is_extreme_weather:
                    threshold = daily_load

                pred_peak.append(False)

            # print(peak_day_picked)
            # monthly_df["pred_peak"] = pred_peak[-num_day_in_month:]


            # print("RECALL")
            # r = monthly_df[(monthly_df["pred_peak"]==True) & (monthly_df["gt_peak"]==True)].shape[0]/N
            # recall.append(r)
            # print(r)
            # print(monthly_df[monthly_df["gt_peak"]==True][["pred_peak","gt_peak"]])

        # print(pred_peak)
        # print(len(pred_peak))
        # print(sum(pred_peak))
        # print(np.mean(recall))

        # print("gt peak")
        # print(sum(gt_peak))
        # tn, fp, fn, tp = confusion_matrix(gt_peak, pred_peak).ravel()
        # print(tn, fp, fn, tp)
        # precision = tp/(tp+fp)
        # recall = tp/(tp+fn)
        # acc = (tp+tn)/(tn + fp + fn + tp)
        # print(precision)
        # print(recall)
        # print(acc)
        # print("#"*20)
        return pred_peak


    def check_extreme_weather(self, today_temp, month_temp, quant=0.9):
        if today_temp < month_temp["MaxTemp"].quantile(1-quant):
            return True
        if today_temp > month_temp["MaxTemp"].quantile(quant):
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

    
    def train(self, X_train, y_train):
        # refactor from probabilistic model
        # create a model
        pass

    def predict(self, X):
        # refactor from probabilistic model
        # predict peak or not peak
        pass
    # p, r, a = probabilistic_model(y_pred_month, y_true, daily_df, N=5, look_ahead=3, quantile=0.9, threshold_p=0.1, delta=0)
    