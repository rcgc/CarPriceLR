import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=CAR+DETAILS+FROM+CAR+DEKHO.csv
from numpy.core.shape_base import block

df = pd.read_csv('carData.csv')

# Getting features
df_X = df['Kms_Driven']
df_Y = df['Selling_Price']*1000

# Kilometers driven
df_X_train = np.array(df_X[: 240])     # 000 - 239
df_X_test = np.array(df_X[240:])      # 240 - 299

# Selling price
df_Y_train = np.array(df_Y[: 240])     # 000 - 239
df_Y_test = np.array(df_Y[240:])      # 240 - 299

# Adding title and axis names
plt.figure('Predicting car price using linear regression')
plt.title('Car price prediction')
plt.xlabel('Kilometers driven')
plt.ylabel('Price')

plt.scatter(df_X_train, df_Y_train, alpha=0.3)

# Adding the column for independent term at the beginning
df_X_train = np.array([np.ones(240), df_X_train]).T

# B = ((X^T) X)^-1 (X^T) (Y)
B = np.linalg.inv(df_X_train.T @ df_X_train) @ df_X_train.T @ df_Y_train

plt.plot([4, 220000], [B[0] + B[1] * 4, B[0] + B[1] * 220000], c="red")

# Mean of the whole dataset (300 instances)
mean = sum(df_Y) / len(df_Y)

print('Independent term: ', B[0])
print('Slope: ', B[1])
print('Mean (whole dataset): $', mean, "usd")

Y_predict_train = list()

for i in df_X_train.T[1]:
    Y_predict_train.append(B[0] + B[1] * i)

Y_predict_train = np.array(Y_predict_train)

mse_train = 0

for i, j in zip(df_Y_train, Y_predict_train):
    mse_train = mse_train + (i - j) * (i - j)

mse_train = mse_train / 300

print(df_X_train.T[1])
print('mse in training: ', mse_train)

ss_res_train = 0

for i in df_Y_train:
    ss_res_train = ss_res_train + abs(i - mean) * abs(i - mean)

ss_tot_train = mse_train * 300

R_train = 1 - (ss_res_train / ss_tot_train)

print('coeficient of determination in training: ', R_train)

Y_predict_test = list()

for i in df_X_test:
    Y_predict_test.append(B[0] + B[1] * i)

Y_predict_test = np.array(Y_predict_test)
mse_test = 0

for i, j in zip(df_Y_test, Y_predict_test):
    mse_test = mse_test + (i - j) * (i - j)

mse_test = mse_test / 300

print('mse in testing: ', mse_test)

ss_res_test = 0

for i in df_Y_test:
    ss_res_test = ss_res_test + abs(i - mean) * abs(i - mean)

ss_tot_test = mse_test * 300

R_test = 1 - (ss_res_test / ss_tot_test)

print('coeficient of determination in testing: ', R_test)

plt.show(block=False)

# X -> input, Y -> predicted
X_queries = []
Y_queries = []

option=0

while(option==0):
    print('0) Make paredictions')
    print('1) Exit')

    option = int(input())

    if(option==1):
        break

    n = int(input("How many integers : "))

    # Queries
    for i in range(0, n):
        ele = int(input())
        print(ele, '->', (B[0] + B[1] * ele)/1000)
        X_queries.append(ele)
        Y_queries.append(B[0] + B[1] * ele)

    plt.scatter(X_queries, Y_queries, c='red', alpha=0.3)
    plt.show(block = False)

    X_queries.clear()
    Y_queries.clear()

plt.show()

# Examples of real queries0
# Kms_driven -> Selling_Price
# 27000 -> 3.35
# 43000 -> 4.75
# 42450 -> 4.6