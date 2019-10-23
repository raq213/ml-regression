# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm

# uploading the data set
dataset = pd.read_excel('dataset/Folds5x2_pp.xlsx', index_col=None)
# Naming each column
dataset.columns =["Amb_temp", "Exhaust_ac", "Amb_press", "Rel_hum", "El_ene_out"]

# ploting the graphs - each variable vs output
fig, axs = plt.subplots(1, 4, sharey=True)
dataset.plot(kind='scatter', x='Amb_temp', y='El_ene_out', ax=axs[0], figsize=(16, 8))
dataset.plot(kind='scatter', x='Exhaust_ac', y='El_ene_out', ax=axs[1])
dataset.plot(kind='scatter', x='Amb_press', y='El_ene_out', ax=axs[2])
dataset.plot(kind='scatter', x='Rel_hum', y='El_ene_out', ax=axs[3])

# creating fitted model
lm = sm.ols(formula='El_ene_out ~ Amb_temp', data=dataset).fit()
# printing coefficients
print(lm.params)
# conclusion: increase of the Ambient temperature leads to deacreasing output power

# creating x parameters with min and max value of temperature
X_new = pd.DataFrame({'Amb_temp':[dataset.Amb_temp.min(), dataset.Amb_temp.max()]})
X_new.head()
# predictions of new output based on min and max points
predicts = lm.predict(X_new)

# ploting the data with the least square line
dataset.plot(kind='scatter', x='Amb_temp', y='El_ene_out')
plt.plot(X_new, predicts, c='red', linewidth=1)



"""

@author: Marcin
"""

