import numpy as np
import pandas as pd

# # Dummy data
# x1 = np.array([60, 62, 67, 70, 71, 72, 75, 78])
# x2 = np.array([22,  25, 24, 20, 15, 14, 14, 11])
# y = np.array([140, 155, 159, 179, 192, 200, 212, 215])

# Read the CSV file
df = pd.read_csv('bmi.csv')

# Extracting the relevant columns
x1 = df['Height'].values
x2 = df['Weight'].values
y = df['Index'].values

n = len(x1)

x1_squared_sum = np.sum(x1 ** 2)
x1_meann = np.mean(x1)
x2_squared_sum = np.sum(x2 ** 2)
x2_meann = np.mean(x2 ** 2)
x1_y_sum = np.sum(x1 * y)
x2_y_sum = np.sum(x2 * y)
x1_x2_sum = np.sum(x1 * x2)

x1_squared_regSum = x1_squared_sum - (np.sum(x1) ** 2 / n)
x2_squared_regSum = x2_squared_sum - (np.sum(x2) ** 2 / n)
x1_y_regSum = x1_y_sum - (np.sum(x1) * np.sum(y) / n)
x2_y_regSum = x2_y_sum - (np.sum(x2) * np.sum(y) / n)
x1_x2_regSum = x1_x2_sum - (np.sum(x1) * np.sum(x2) / n)

b1 = (x2_squared_regSum * x1_y_regSum - x1_x2_regSum * x2_y_regSum) / \
    (x1_squared_regSum * x2_squared_regSum - x1_x2_regSum ** 2)
b2 = (x1_squared_regSum * x2_y_regSum - x1_x2_regSum * x1_y_regSum) / \
    (x1_squared_regSum * x2_squared_regSum - x1_x2_regSum ** 2)
b0 = np.mean(y) - b1*np.mean(x1) - b2*np.mean(x2)

print(f"Multi-linear Regression Equation: {b0:.3f} + {b1:.3f}x1 + {b2:.3f}x2")

# Predictions using the calculated coefficients
predictions = np.round(b0 + b1 * x1 + b2 * x2, 2)

# Compare predictions with actual results
comparison_df = pd.DataFrame({'Actual': y, 'Predicted': predictions})
print(comparison_df)

# Calculate Mean Squared Error (MSE) as a measure of the model's performance
mse = np.mean((predictions - y) ** 2)
print(f"Mean Squared Error (MSE): {mse:.3f}")

# User inputs for x1 and x2
user_input_x1 = float(input("Enter the value for x1 (Height): "))
user_input_x2 = float(input("Enter the value for x2 (Weight): "))

# Prediction for the user inputs
user_prediction = b0 + b1 * user_input_x1 + b2 * user_input_x2

# Round the user prediction to one decimal place
rounded_user_prediction = round(user_prediction, 1)

print(
    f"Predicted Index for x1={user_input_x1}, x2={user_input_x2}: {rounded_user_prediction}")
