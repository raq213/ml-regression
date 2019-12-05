# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.graphics.correlation import plot_corr

# uploading the data set
dataset = pd.read_excel("Folds5x2_pp.xlsx", index_col=None)

# Naming each column
dataset.columns =["Amb_temp", "Exhaust_ac", "Amb_press", "Rel_hum", "El_ene_out"]
dataset = dataset.head(1000)
# ploting the graphs - each variable vs output
fig1, axs = plt.subplots(1, 4, sharey=True)
dataset.plot(kind='scatter', x='Amb_temp', y='El_ene_out', ax=axs[0], figsize=(16, 8))
dataset.plot(kind='scatter', x='Exhaust_ac', y='El_ene_out', ax=axs[1])
dataset.plot(kind='scatter', x='Amb_press', y='El_ene_out', ax=axs[2])
dataset.plot(kind='scatter', x='Rel_hum', y='El_ene_out', ax=axs[3])

# My input variables
X = dataset[['Amb_temp', 'Rel_hum']] 
# My output variables
y = dataset['El_ene_out']

# adding an intercept to the model (statmodels doesn not add it automaticly)
Xstat = sm.add_constant(X)

# creating fitted model based on Ordinary Least Squares
model = sm.OLS(y, Xstat).fit()
print(model.summary())
# CONCLUSION: all of the variables have impact on the output

# predicting output
predictions = model.predict(Xstat)
pred = model.fittedvalues.copy()
true = dataset['El_ene_out']
residual = true - pred

# partial regression graphs to visualise trend lines
# model as the first parameter, then variable I want to analyze
fig2 = plt.figure(figsize=(20,12))
fig2 = sm.graphics.plot_partregress_grid(model, fig=fig2)

# The “Y and Fitted vs. X” graph plots the dependent variable against predicted values 
# with a confidence interval
fig3 = plt.figure(figsize=(15,8))
fig3 = sm.graphics.plot_regress_exog(model,'Amb_temp', fig=fig3)

#seaborn
g = sns.pairplot(dataset)
corr = dataset[:-1].corr()
fig4 = plot_corr(corr,xnames=corr.columns)

# sklearn part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})



"""

@author: Marcin
"""

