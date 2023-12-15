import pandas as pd
import numpy as np

learning_rate = 0.01
epochs = 1000
df = pd.read_csv('BMI.csv')

x1 = df['Height'].values
x2 = df['Weight'].values
y = df['Index'].values


def normalize_data(data):
    return (data - data.mean()) / data.std()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit(x1, x2, y, epochs):
    X = np.column_stack((x1, x2))
    weights = np.zeros(X.shape[1])
    bias = 0
    sse_list = []  # List to store SSE at each iteration

    for epoch in range(epochs):
        sse = 0  # Initialize SSE for the current epoch
        for i in range(len(df)):
            z = np.dot(X[i], weights) + bias
            y_pred = sigmoid(z)

            # Update weights and bias
            weights = weights + learning_rate * (y[i] - y_pred) * X[i]
            bias = bias + learning_rate * (y[i] - y_pred)

            # Calculate SSE for the current data point and accumulate
            sse += 0.5 * (y[i] - y_pred) ** 2

        # Append MSE for the current epoch to the list
        sse_list.append(sse / len(df))

        # Check if the current SSE is the same as the previous SSE
        if epoch > 0 and sse_list[-1] == sse_list[-2]:
            break

    print(f"Converged at iteration {epoch + 1}. SSE = {sse_list[-1]}")
    return weights, bias, sse_list


norm_x1 = normalize_data(x1)
norm_x2 = normalize_data(x2)

weights, bias, sse_list = fit(norm_x1, norm_x2, y, epochs)

# Print the SSE for each iteration
for epoch, sse in enumerate(sse_list):
    print(f"Iteration {epoch + 1}: SSE = {sse}")

# Assuming you have new values for height and weight
new_height = 170
new_weight = 70

# Normalize the new data using the same normalization function
normalized_new_height = (new_height - x1.mean()) / x1.std()
normalized_new_weight = (new_weight - x2.mean()) / x2.std()

# Combine the normalized values into a single array
new_data = np.array([normalized_new_height, normalized_new_weight])

# Make a prediction using the trained weights and bias
prediction = sigmoid(np.dot(new_data, weights) + bias)

print("Predicted Probability:", prediction)
