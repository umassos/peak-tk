
import pandas as pd
import numpy as np

class ExtremeTemperatureModel():
    def __init__(self, isSummerMonth=None, max_peakday=100):
        self.monthly_threshold = []
        # TODO: check length of isSummerMonth
        if isSummerMonth is None:
            self.isSummer = [False,False,False,False,True,True,True,True,True,False,False,False]
        else:
            self.isSummer = isSummerMonth

        self.max_peakday = max_peakday
        self.reset_monthly(-1)

    def reset_monthly(self, new_month):
        self.peakday_picked = 0
        self.current_month = new_month

    def fit(self, historical_df, quantile=0.9):
        # expected input
        # historical_df has index as datetime
        # columns: temperature

        self.monthly_threshold = []
        for i in range(1, 13):

            quant = quantile
            if not self.isSummer[i-1]:
                quant = 1-quantile

            month_mask = historical_df.index.month == i

            temp_threshold = historical_df[month_mask]["temperature"].quantile(quant)

            self.monthly_threshold.append(temp_threshold)


    def predict(self, X):
        # X is a list of list
        # nested list contains
        # [0] - temperature
        # [1] - month

        # check temp list length == month list length
        y_hat = []
        for x in X:
            temp = x[0]
            m = x[1]

            if m != self.current_month:
                self.reset_monthly(m)

            if self.isSummer[m-1]:
                if temp > self.monthly_threshold[m-1]:
                    if self.peakday_picked >= self.max_peakday:
                        y_hat.append(False)
                    else:
                        y_hat.append(True)
                        self.peakday_picked += 1
                else:
                    y_hat.append(False)
            else:
                if temp < self.monthly_threshold[m-1]:
                    if self.peakday_picked >= self.max_peakday:
                        y_hat.append(False)
                    else:
                        y_hat.append(True)
                        self.peakday_picked += 1
                else:
                    y_hat.append(False)

        return y_hat


# # 1.1 FIND MONTHLY PEAK 
# def extreme_temperature_model(historical_, k=5, quantile=0.8, isSummer=None, debug=True): 
#     temp_option = "AvgTemp"
# #     temp_option = "MaxTemp"
#     # set season, winter=>cold=>more energy usage, summer=>hot=>more energy usage
#     if isSummer is None:
#         isSummer = [False,False,False,False,True,True,True,True,True,False,False,False]
#     # iterate one month at a time range(1,13)
#     p = []
#     r = []
#     for i in range(1, 13):
#         # flip quantile for winter
#         quant = quantile
#         if not isUpper[i-1]:
#             quant = 1-quantile
#         # filter
#         month_mask = training_df.index.month == i
#         month_mask_test = testing_df.index.month == i
#         # plot correlation
#         if debug:
#             testing_df[month_mask_test].plot(x=temp_option, y='MaxPower', style='o', title="Testing data's correlation: " + month_name[i])
#             training_df[month_mask].plot(x=temp_option, y='MaxPower', style='o', title="Training data's correlation: " + month_name[i])
#             plt.show()
#             # plot probability distribution
#             training_df[month_mask][temp_option].plot(kind='kde'
#                                                             , title="Training data dist for: " + month_name[i], legend=True ,label="Train")
#             ax = testing_df[month_mask_test][temp_option].plot(kind='kde'
#                                                                 , title="Data dist for: " + month_name[i], legend=True, label="Test")
#             ax.set_xlabel("Temperature (F)")
#             plt.show()

#         # define (adjustable) threshold
#         threshold = training_df[month_mask][temp_option].quantile(quant)
#         if debug:
#             print("quantile: {}".format(quant))
#             print("threshold: {}".format(threshold))

#         # set monthly peak to false
#         testing_df.loc[month_mask_test,"MonthlyPeak"] = False
#         testing_df.loc[month_mask_test,"PredictedMonthlyPeak"] = False

#         # find groundtruth monthly peak in testing
#         testing_df.loc[month_mask_test & (testing_df.index.isin(testing_df[month_mask_test].nlargest(k,'MaxPower').index)),"MonthlyPeak"] = True

#         # predict monthly peak in testing from threshold
#         if isUpper[i-1]:
#             testing_df.loc[(month_mask_test) & (testing_df[temp_option] > threshold),"PredictedMonthlyPeak"] = True
#         else:
#             testing_df.loc[(month_mask_test) & (testing_df[temp_option] < threshold),"PredictedMonthlyPeak"] = True
            
#         # add rank (use 'first', first appears in the list)
#         testing_df.loc[month_mask_test, "PeakRank"] = testing_df.loc[month_mask_test, "MaxPower"].rank(method='first', ascending=False)
# #         print(testing_df.loc[(month_mask) & (testing_df["MaxTemp"] > threshold)])

#         # evaluate result.
#         num_row = testing_df[month_mask_test].shape[0]
        
#         groundtruth_df = testing_df[(month_mask_test) & (testing_df["MonthlyPeak"]==True)]
#         predict_df = testing_df[(month_mask_test) & (testing_df["PredictedMonthlyPeak"]==True)]
#         correct_df = testing_df[(month_mask_test) & (testing_df["MonthlyPeak"]==True) & (testing_df["PredictedMonthlyPeak"]==True)]
#         if debug:
#             print("number of day in month: {}".format(num_row))
#             if k != groundtruth_df.shape[0]:
#                 print("Error, k != number of k-peak day")
#             print("number of k peak day: {}".format(groundtruth_df.shape[0]))
#             print("number of predicted peak day: {}".format(predict_df.shape[0]))
#             print("number of correct predicted: {}".format(correct_df.shape[0]))
#             # how many top k are included? starting from highest peak
#             print(correct_df["PeakRank"])
#             print(predict_df[["PeakRank",temp_option]])
        

#         tn, fp, fn, tp = confusion_matrix(testing_df[month_mask_test]["MonthlyPeak"], testing_df[month_mask_test]["PredictedMonthlyPeak"]).ravel()
#         if debug:
#             print(tn, fp, fn, tp)
#             print("#"*50)

#         precision = tp/(tp+fp)
#         recall = tp/(tp+fn)

#         p.append(precision)
#         r.append(recall)

        
        
#     print(k)
#     print("precision")
#     p = np.array(p)
#     p[np.isnan(p)] = 0
#     print(p)
#     print(np.mean(p))
#     print("recall")
#     print(r)
#     print(np.mean(np.array(r)[[0,1,5,6,7,10,11]]))
#     if debug:
#         plot_precision_recall(p, r)
#     print("#"*50)
        

# plt.rcParams['figure.figsize'] = [4, 4]
# find_monthly_peak(training_df, testing_df, 10, 0.8)
    