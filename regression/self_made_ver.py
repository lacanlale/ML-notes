print("      (.)~(.)")
print("     (-------)")
print("----ooO-----Ooo----")
print("SELFMADE REGRESSION")                                       
print("-------------------")
print("     ( )   ( )")
print("     /|\   /|\\")

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import style
from statistics import mean

style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(size, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(size):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def find_best_fit_slope_and_intercept(xs, ys):
    numerator = (mean(xs) * mean(ys)) - (mean(xs * ys))
    denominator = (mean(xs) * mean(xs)) - (mean(xs * xs))
    m = numerator/denominator

    b = mean(ys) - (m*mean(xs))
    return m, b


def find_squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)


# cod = Coefficient of Determination
def find_cod(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = find_squared_error(ys_orig, ys_line)
    squared_error_y_mean = find_squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


xs, ys = create_dataset(size=40, variance=40, step=2, correlation='pos')

m, b = find_best_fit_slope_and_intercept(xs, ys)
regression_line = [(m*x) + b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b
 
r_squared = find_cod(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()