# -*- coding=utf8 -*-
import numpy as np
from keras import backend as K


# 损失函数（交叉损失）
def mean_negative_log_probs(y_true, y_pred):
    log_probs = -K.log(y_pred)
    log_probs *= y_true
    return K.sum(log_probs) / K.sum(y_true)


# 准确度函数
def compute_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # 计算真值1且预测1
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  # 预测总数
    precision = true_positives / (predicted_positives + K.epsilon())  # K.epsilon()：极小量
    return precision


# 召回率计算
def compute_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # 计算真值1且预测1
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))  # 真值总数
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def array_del(arr):
    # np.de1
    pass

if __name__ == '__main__':
    pass