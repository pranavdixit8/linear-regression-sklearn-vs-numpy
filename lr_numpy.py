from numpy import *
import pandas as pd
import matplotlib.pyplot as plt





def compute_squared_mean_error(b, m , bmi_life_data):

	total_error = 0

	Y = bmi_life_data[['Life expectancy']]
	N = float(len(Y))

	for i in range(0, len(Y)):

		x = bmi_life_data.loc[i, 'BMI']
		y = bmi_life_data.loc[i,'Life expectancy']

		total_error+=(y - (m*x + b))**2 

	return total_error/N



def step_gradient(current_m, current_b, bmi_life_data, learning_rate):

	b_gradient = 0
	m_gradient = 0

	Y = bmi_life_data[['Life expectancy']]

	N = float(len(Y))

	for i in range (0,len(Y)):
		x = bmi_life_data.loc[i, 'BMI']
		y = bmi_life_data.loc[i,'Life expectancy']

		b_gradient+= -(2/N) * ( y - (current_m*x + current_b))
		m_gradient+= -(2/N) * x * ( y - (current_m*x + current_b))

	new_m = current_m - (learning_rate*m_gradient)
	new_b = current_b - (learning_rate * b_gradient)
	

	return [new_m, new_b]


def gradient_descent(bmi_life_data, initial_m, initial_b, learning_rate, num_iteration):

	m = initial_m
	b = initial_b

	for i in range(num_iteration):

		m,b = step_gradient(m, b , bmi_life_data , learning_rate)

	return [m,b]


def predict(b,m, x):

	return m*x + b



def lr_numpy():

	bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
	
	initial_m = 0
	initial_b = 0


	num_iteration  =  2000
	learning_rate = 0.0001

	print("Running gradient Descent:")

	print("Starting  with b = {}, m = {}, squared mean error = {}".format(initial_b, initial_m, compute_squared_mean_error(initial_b, initial_m, bmi_life_data)))

	[m,b] =  gradient_descent(bmi_life_data, initial_m, initial_b, learning_rate, num_iteration)

	print("After running  {} iterations,  b = {}, m = {}, squared mean error = {}".format(num_iteration, b, m, compute_squared_mean_error(b, m, bmi_life_data)))
	
	predict_life_exp = predict(b, m , bmi_life_data[['BMI']].values)

	plt.scatter(bmi_life_data[['BMI']][0:-20], bmi_life_data[['Life expectancy']][0:-20],  color='black')
	plt.plot(bmi_life_data[['BMI']], predict_life_exp , color='blue', linewidth=3)
	plt.show()


if __name__ == "__main__":
	lr_numpy()