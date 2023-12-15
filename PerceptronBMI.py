import random
import pandas as pd
import numpy as np

random.seed(random.randint(0, 10000000000000))


def getHyperparameters():
    if input("Do you want to randomize parameter (y|n)? ") == "y":
        print("Randomizing parameters...")
        alpha = random.randint(0, 10) / 10
        theta = random.randint(0, 10) / 10
        w1 = random.randint(-10, 10) / 10
        w2 = random.randint(-10, 10) / 10
        print("Alpha: ", alpha)
        print("Theta: ", theta)
        print("w1: ", w1)
        print("w2: ", w2)

        return alpha, theta, w1, w2

    else:
        return float(input("Enter alpha: ")), float(input("Enter theta: ")), float(input("Enter w1: ")), float(input("Enter w2: "))


def Normalize(data):
    return (data - data.mean()) / data.std(), data.mean(), data.std()


def Denormalize(data, mean, std):
    return data * std + mean


def sigmoid(x):
    # Prevent overflow
    if x < -500:
        return 0
    else:
        return 1 / (1 + np.exp(-x))


def predict(x1, x2, w1, w2, theta):
    return sigmoid(w1 * x1 + w2 * x2 + theta)


def plateauCheck(SSE, prevSSE):
    if SSE == prevSSE:
        return True
    else:
        return False


def Perceptron(x1, x2, y, alpha, theta, w1, w2):
    SSEThreshold = float(input("Enter SSE Threshold: "))
    SSE = SSEThreshold + 1
    prevSSE = 0
    j = 1
    while SSE > SSEThreshold:
        prevSSE = SSE
        SSE = 0
        for i in range(len(x1)):
            yHat = predict(x1[i], x2[i], w1, w2, theta)
            e = y[i] - yHat
            SSE += e ** 2
            w1 = w1 + alpha * x1[i] * e
            w2 = w2 + alpha * x2[i] * e
            theta = theta - alpha * e

        if plateauCheck(SSE, prevSSE):
            print("Plateau reached. Exiting...")
            break
        print("Iteration: ", j)
        j += 1
        print("SSE: ", SSE)
        print()

    return w1, w2, theta


def main():
    BMI = pd.read_csv('BMI.csv')
    alpha, theta, w1, w2 = getHyperparameters()
    x1, meanX1, stdX1 = Normalize(BMI['Height'])
    x2, meanX2, stdX2 = Normalize(BMI['Weight'])
    y, meanY, stdY = Normalize(BMI['Index'])
    w1, w2, theta = Perceptron(x1, x2, y, alpha, theta, w1, w2)
    print("Training Complete. Final Parameters:")
    print("w1: ", w1)
    print("w2: ", w2)
    print("theta: ", theta)
    print()

    print("Testing...")
    while True:
        x1 = float(input("Enter Height: "))
        x2 = float(input("Enter Weig    ht: "))
        yHat = predict((x1 - meanX1) / stdX1,
                       (x2 - meanX2) / stdX2, w1, w2, theta)
        print("BMI Index: ", Denormalize(yHat, meanY, stdY))
        print()
        if input("Do you want to continue (y|n)? ") == "n":
            break


if __name__ == '__main__':
    main()
