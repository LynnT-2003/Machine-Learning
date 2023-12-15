import random
import pandas as pd
import numpy as np


learning_rate = 0.01
epochs = 1000
file_path = 'BMI.csv'
pd_data = pd.read_csv('BMI.csv')


def normalize_data(data):
    # Exclude the 'Gender' column from normalization
    return (data.iloc[:, 1:-1] - data.iloc[:, 1:-1].mean()) / data.iloc[:, 1:-1].std()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_perceptron(data, learning_rate, epochs):
    # Exclude the target column and 'Gender'
    num_features = len(data.columns) - 2
    weights = [random.uniform(-1, 1) for _ in range(num_features)]

    for epoch in range(epochs):
        for _, row in data.iterrows():
            # Exclude the 'Gender' column from features
            features = row.iloc[1:-1]
            target = row.iloc[-1]

            # Calculate the weighted sum
            weighted_sum = sum(weights[i] * features.iloc[i]
                               for i in range(num_features))

            # Update weights based on the perceptron learning rule
            for i in range(num_features):
                weights[i] += learning_rate * \
                    (target - int(weighted_sum > 0)) * features.iloc[i]

    return weights


def predict_bmi(features, weights):
    # Calculate the weighted sum
    weighted_sum = sum(weights[i] * features.iloc[i]
                       for i in range(len(features)))
    predicted_probability = sigmoid(weighted_sum)
    return predicted_probability


normalized = normalize_data(pd_data)
print(normalized)

# print(pd_data)
weights_perceptron = train_perceptron(pd_data, learning_rate, epochs)
print(weights_perceptron)

# Example usage for prediction
# Replace with the actual features of a new data point
# Example new data point without 'Gender'

new_data_point = pd.Series([174.0, 9000.0])
predicted_bmi = predict_bmi(new_data_point, weights_perceptron)
print("Predicted BMI Index:", predicted_bmi)
