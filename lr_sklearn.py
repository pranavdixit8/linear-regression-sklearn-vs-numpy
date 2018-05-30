
import pandas as pd
from sklearn import linear_model
from numpy import *
import matplotlib.pyplot as plt

bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

lr_model = linear_model.LinearRegression()
lr_model.fit(bmi_life_data[['BMI']][0:-20], bmi_life_data[['Life expectancy']][0:-20])


predict_life_exp = lr_model.predict(bmi_life_data[['BMI']])

plt.scatter(bmi_life_data[['BMI']][0:-20], bmi_life_data[['Life expectancy']][0:-20],  color='black')
plt.plot(bmi_life_data[['BMI']], predict_life_exp , color='blue', linewidth=3)

plt.xlabel("BMI")
plt.ylabel("Life expectancy")

plt.show()

