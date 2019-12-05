# Program in its current state is prepared to make a single variable polinomial regression on Folds5x2_pp.xlsx dataset in dataset folder in this repository.

import operator

import numpy as np
import matplotlib.pyplot as plt
import pandas

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# # Bike Y
# colnames = ["instant","dteday","season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual","registered","cnt"]
# df = pandas.read_csv("C:/git/machine-learning/ML_Regression/Bike-Sharing-Dataset/day.csv", sep=',', names=colnames)

# Concrete
# colnames = ['cement', 'slag', 'ash', 'water', 'plasticizer', 'coarse_aggregate', 'fine_aggregate', 'age', 'str']
# df = pandas.read_csv('C:/git/machine-learning/ML_Regression/data/Concrete_Data.csv', sep=';', names=colnames)

# Marcin
colnames =["Amb_temp", "Exhaust_ac", "Amb_press", "Rel_hum", "El_ene_out"]
df = pandas.read_excel("C:\git\machine-learning\ML_Regression\dataset\Folds5x2_pp.xlsx", names = colnames)

electricEnergyOut = df.El_ene_out.to_numpy()

ambientTemperature = df.Amb_temp.to_numpy()
exhaust = df.Exhaust_ac.to_numpy()
ambientPressure = df.Amb_press.to_numpy()
realHumidity = df.Rel_hum.to_numpy()


def PolynomialReression(degree, X,XName, Y, YName,toPlot):

    # transforming the data to include another axis
    X = X[:, np.newaxis]
    Y = Y[:, np.newaxis]

    polynomial_features= PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(X)

    model = LinearRegression()
    model.fit(x_poly, Y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
    r2 = r2_score(Y,y_poly_pred)

    if toPlot:
        print(rmse)
        print(r2)
        plt.scatter(X, Y, s=10)
        # sort the values of x before line plot
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
        X, y_poly_pred = zip(*sorted_zip)
        
        title = YName + " (" + XName + ")"
        plt.xlabel(XName)
        plt.ylabel(YName)
        plt.title(title)
        plt.plot(X, y_poly_pred, color='r')
        plt.show()

    return r2


# Make a plot of dependency between degree of polynomial and how well it feats the dataset
r2 = []
degrees = 50
for degree in range(degrees):
    r2.append(PolynomialReression(degree,ambientTemperature, "Ambient Temperature",electricEnergyOut, "Output Electric Energy", False))

x_r2 = range(degrees)
y_r2 = r2
plt.plot(x_r2, y_r2)
plt.show()
best_degree = y_r2.index(max(y_r2))
print()

# Ploting best rgression line 
PolynomialReression(best_degree,ambientTemperature, "Ambient Temperature",electricEnergyOut, "Output Electric Energy", True)