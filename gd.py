import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# stores the errors/loss for visualisation
__errors__ = []


def hypothesis(params, data):

    acum = 0
    for i in range(len(params)):
        # evaluates h(x) = a+bx1+cx2+ ... nxn..
        acum = acum + params[i] * data[i]

    return acum


def show_errors(params, data, y):

    global __errors__
    error_acum = 0

    for i in range(len(data)):
        hyp = hypothesis(params, data[i])
        # print("hyp  %f  y %f " % (hyp, y[i]))
        error = hyp - y[i]

        # this error is the original cost function,
        # (the one used to make updates in GD is the derivated verssion of this formula)
        error_acum = +error ** 2

    mean_error_param = error_acum / len(data)
    __errors__.append(mean_error_param)


def GD(params, data, y, alfa):

    temp = list(params)
    general_error = 0
    for j in range(len(params)):
        acum = 0;
        error_acum = 0
        for i in range(len(data)):
            error = hypothesis(params, data[i]) - y[i]
            # Summatory part of the Gradient Descent formula for linear Regression.
            acum = acum + error * data[i][j]

        # Subtraction of original parameter value with learning rate included.
        temp[j] = params[j] - alfa * (1 / len(data)) * acum
    return temp


def scaling(samples):

    acum = 0
    samples = np.asarray(samples).T.tolist()
    for i in range(1, len(samples)):
        for j in range(len(samples[i])):
            acum = acum + samples[i][j]
        avg = acum / (len(samples[i]))
        max_val = max(samples[i])

        # print("avg %f" % avg)
        # print(max_val)

        for j in range(len(samples[i])):
            # print(samples[i][j])
            # Mean scaling
            samples[i][j] = (samples[i][j] - avg) / max_val

    return np.asarray(samples).T.tolist()


df = pd.read_csv('carData.csv')

df_X = df['Kms_Driven']
df_Y = df['Selling_Price'] * 1000

# [theta, bias]
params = [0, 0]

data = df_X.values.tolist()
y = df_Y.values.tolist()

# Learning rate
alfa = 0.01

for i in range(len(data)):
    if isinstance(data[i], list):
        data[i] = [1] + data[i]
    else:
        data[i] = [1, data[i]]

data = scaling(data)

epochs = 0

# run gradient descent until local minima is reached
while True:
    oldparams = list(params)
    # print(params)
    params = GD(params, data, y, alfa)

    # only used to show errors, it is not used in calculation
    show_errors(params, data, y)
    # print(params)
    epochs = epochs + 1

    # local minima is found when there is no further improvement
    if oldparams == params or epochs == 2000:
        # print("samples:")
        # print(samples)

        print("final params:")
        print(params)
        break


print(__errors__[1999])
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.plot(__errors__)
plt.show()