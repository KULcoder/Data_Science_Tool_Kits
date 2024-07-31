import numpy as np
from sklearn.metrics import roc_auc_score

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def true_positive(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 1))

def false_positive(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 1))

def false_negative(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 0))

def true_negative(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 0))

def recall(y_true, y_pred):
    # proportion of actually positive instances that are correctly identified
    # out of all positives, what percentage we found?
    # important if finding all positives is very crucial, like health condition
    true_positive = true_positive(y_true, y_pred)
    actual_positive = np.sum(y_true == 1)
    return true_positive / actual_positive if actual_positive != 0 else 0

def precision(y_true, y_pred):
    # how accurate is your prediction?
    # out of all predicted positives, what percentage are really true?
    true_positive = true_positive(y_true, y_pred)
    predicted_positive = np.sum(y_pred == 1)
    return true_positive / predicted_positive if predicted_positive != 0 else 0

def r2(y_true, y_pred):
    """
    Coefficient of Determination: is the proportion of the variation in the dependent 
    variable that is predictable from the independent variable(s).

    Range from 0 to 1. Closer it is to 1, betther the linear fit is.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def f1(y_true, y_pred):
    # the harmonic mean of precision and recall
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

def specificity(y_true, y_pred):
    # true negative rate: 
    # the proportion of actual negatives that were identified correctly
    true_negative = true_negative(y_true, y_pred)
    actual_negative = np.sum(y_true == 0)
    return true_negative / actual_negative if actual_negative != 0 else 0

def auc(y_true, y_pred):
    # area under ROC (receiver operating characteristics) curve,
    # roc showing performance of a classification model at all classification thres
    # larger the better
    return roc_auc_score(y_true, y_pred)