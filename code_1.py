# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('Salary_Data.csv') 

# First five records of the dataset
print(data.head())

# Dataset information
print(data.info())


# Define features and target
X = data[['YearsExperience']]  # Corrected to the actual column name
y = data['Salary']  # Corrected to the actual target column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict using Linear Regression model
y_pred_lr = lr_model.predict(X_test)

# Calculate MSE for Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr}")
# Predict the salary for an employee with 6 years of experience
years_of_experience = pd.DataFrame([[6]], columns=['YearsExperience'])  # Use a DataFrame
predicted_salary = lr_model.predict(years_of_experience)

print(f"The predicted salary for an employee with 6 years of experience is: {predicted_salary[0]:.2f}")


# Plotting the actual vs predicted values
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.plot(X_test, y_pred_lr, color='blue', label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Actual vs Predicted Salary')
plt.legend()
plt.show()
