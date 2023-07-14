#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

wind_data = pd.read_excel(r"...\Mali wind farm.xlsx")
wind_data_values_ndarray = wind_data.values
wind_data_values = np.array(wind_data_values_ndarray)
[m, n] = wind_data_values.shape
for ii in range(3, n):
    print(ii)
    wind_power = wind_data_values[:, ii]
    series = wind_power
    # series = series - np.mean(series)  # Centralization(option)
    # step1 Embedding
    windowLen = 20  
    seriesLen = len(series)  
    K = seriesLen - windowLen + 1
    X = np.zeros((windowLen, K))
    for i in range(K):
        X[:, i] = series[i:i + windowLen]

    # step2:Decompositon
    U, sigma, VT = np.linalg.svd(X, full_matrices=False)

    sum_sum = 0
    weight_value = 0
    sigma_number = 0
    for i in range(VT.shape[0]):
        sum_sum += sigma[i]
        weight_value = sum_sum / sum(sigma)
        print(weight_value)
        sigma_number += 1
        if weight_value >= 0.95:
            break

    for i in range(VT.shape[0]):
        VT[i, :] *= sigma[i]
    A = VT

    # step3:Grouping and Reconstruction
    rec = np.zeros((windowLen, seriesLen))
    for i in range(windowLen):
        for j in range(windowLen - 1):
            for m in range(j + 1):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (j + 1)
        for j in range(windowLen - 1, seriesLen - windowLen + 1):
            for m in range(windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= windowLen
        for j in range(seriesLen - windowLen + 1, seriesLen):
            for m in range(j - seriesLen + windowLen, windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (seriesLen - j)

    # plt.figure()
    # for i in range(20):
    #    ax = plt.subplot(10, 2, i + 1)
    #    ax.plot(rec[i, :])
    # plt.show()

    rrr = np.sum(rec[:sigma_number, :], axis=0)  
    rrr_trans = rrr.reshape((len(rrr), 1))
    wind_data_values[:, ii] = rrr_trans[:, 0]

# transform wind_data_values to pandas DataFrame
a_pd = pd.DataFrame(wind_data_values)
# create writer to write an excel file
writer = pd.ExcelWriter(r'...\mali.xlsx')
# write in ro file, 'sheet1' is the page title, float_format is the accuracy of data
a_pd.to_excel(writer, 'sheet1',header = None,index=False,float_format='%.6f')
# save file
writer.save()
# close writer
writer.close()
