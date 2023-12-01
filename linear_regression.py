weights = [140, 155, 159, 179, 192, 200, 212]
heights = [60, 62, 67, 70, 71, 72, 75]

wh = []
w2 = []
h2 = []

for i in range(len(weights)):
    wh.append(weights[i] * heights[i])

for i in range(len(heights)):
    h2.append(heights[i] * heights[i])

for i in range(len(weights)):
    w2.append(weights[i] * weights[i])

wh_sum = sum(wh)
w2_sum = sum(w2)
h2_sum = sum(h2)

w_sum = sum(weights)
h_sum = sum(heights)

n = len(weights)

b0 = ((h_sum * w2_sum) - (w_sum * wh_sum)) / \
    ((n * w2_sum)-(w_sum * w_sum))

b1 = ((n * wh_sum) - (w_sum * h_sum)) / \
    ((n * w2_sum) - (w_sum ** 2))


def linear_regression(x):
    return (b0 + b1 * x)


pred_x = [140, 155, 159, 179, 192, 200, 212]
pred_y = []

for each in pred_x:
    pred_y.append(round(linear_regression(each), 1))

print(f"Linear Regression: {round(b1, 4)}x + {round(b0, 4)}")
print(f"Predictor Values: {pred_x}")
print(f"Predicted Values: {pred_y}")
