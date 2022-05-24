import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    return np.average(np.abs(np.array(y_true) - np.array(y_pred)))

def mean_squared_error(y_true, y_pred):
    return np.average(np.square(np.array(y_true) - np.array(y_pred)))

def confusion_matrix_(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tn, fp, fn, tp)

def recall(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp/(tp+fn)

def precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp/(tp+fp)

def f1_score(y_true, y_pred):
    r = recall(y_true, y_pred)
    p = precision(y_true, y_pred)
    return 2 * (r*p)/(r+p)

# def sensitivity(y_true, y_pred):
#   tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#   return tp/(tp+fn)

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn/(fp+tn)

def balanced_accuracy(y_true, y_pred):
    return (recall(y_true, y_pred) + specificity(y_true, y_pred))/2



def monthly_peak_day_accuracy(monthly_demand, predicted_result, top_k):

    if len(monthly_demand) != len(predicted_result):
        print("ERROR: monthly_demand ({}) and predicted_result ({}) length must be equal.".format(len(monthly_demand), len(predicted_result)))
        return
        
    temp_df = pd.DataFrame(monthly_demand)
    temp_df.columns = ["gt_demand"]
    temp_df["predicted_peak"] = predicted_result
    temp_df["gt_peak"] = False
    temp_df.loc[(temp_df.index.isin(temp_df.nlargest(top_k,'gt_demand').index)),"gt_peak"] = True

    correct_df = temp_df[(temp_df["predicted_peak"]==True) & (temp_df["gt_peak"]==True)]

    print("correct: {}".format(correct_df.shape[0]/top_k))

    return temp_df
#         

def peak_hour_accuracy(y_true, y_pred, only_peakday=False, num_peak_day=10):

    if y_true.shape[0] != y_pred.shape[0]:
        print("the size of both y should be equal")
        return
    r = []
    p = []
    a = []
    for i in range(y_true.shape[0]):
        y_t = y_true[i]
        y_p = y_pred[i]
        r.append(recall(y_t, y_p))
        p.append(precision(y_t, y_p))
        a.append(balanced_accuracy(y_t, y_p))
        # print(recall(y_t, y_p), precision(y_t, y_p))
    return np.mean(r), np.mean(p), np.mean(a)



