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

Y_predict = list()

for i in df_X_test:
    Y_predict.append(B[0] + B[1] * i)

Y_predict = np.array(Y_predict)
mse = 0

for i, j in zip(df_Y_test, Y_predict):
    mse = mse + (i-j)*(i-j)

mse = mse/300

print('mse: ', mse)

ss_res = 0

for i in df_Y_test:
    ss_res = ss_res + abs(i - mean) * abs(i - mean)

ss_tot = mse * 300

R = 1 - (ss_res / ss_tot)

print('coeficient of determination: ', R)

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
        print(ele, '->', B[0] + B[1] * ele)
        X_queries.append(ele)
        Y_queries.append(B[0] + B[1] * ele)

    plt.scatter(X_queries, Y_queries, c='red', alpha=0.3)
    plt.show(block = False)

    X_queries.clear()
    Y_queries.clear()

plt.show()