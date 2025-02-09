import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def backward_propagation(X, y, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, hidden1_layer_output, hidden2_layer_output, output, learning_rate):

    output_delta = (y - output) * sigmoid_derivative(output)  # (Ytarget - Ypredicted) * y5(1 - y5) => σ5

    hidden2_delta =sigmoid_derivative(hidden2_layer_output) * (output_delta.dot(weights_hidden2_output.T)) # y3(1 - y3) (weight * σ5)

    hidden1_delta = sigmoid_derivative(hidden1_layer_output) * (hidden2_delta.dot(weights_hidden1_hidden2.T))

    weights_hidden2_output += hidden2_layer_output.T.dot(output_delta) * learning_rate
    weights_hidden1_hidden2 += hidden1_layer_output.T.dot(hidden2_delta) * learning_rate # ( y4 * σ5 ) + previous weight
    weights_input_hidden1 += X.T.dot(hidden1_delta) * learning_rate

    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
input_size = 2
hidden1_size = 2
hidden2_size = 2
output_size = 1
learning_rate = 0.5


weights_input_hidden1 = np.random.rand(input_size, hidden1_size)
weights_hidden1_hidden2 = np.random.rand(hidden1_size, hidden2_size)
weights_hidden2_output = np.random.rand(hidden2_size, output_size)

hidden1_layer_input = np.dot(X, weights_input_hidden1)
hidden1_layer_output = sigmoid(hidden1_layer_input)
hidden2_layer_input = np.dot(hidden1_layer_output, weights_hidden1_hidden2)
hidden2_layer_output = sigmoid(hidden2_layer_input)
output_layer_input = np.dot(hidden2_layer_output, weights_hidden2_output)
output = sigmoid(output_layer_input)


weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output = backward_propagation(
    X, y, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, hidden1_layer_output, hidden2_layer_output, output, learning_rate
)

print("Weight: Input to hidden1")
print(weights_input_hidden1)
print("Weight: hidden1 to hidden2")
print(weights_hidden1_hidden2)
print("Weight: hidden2 to output")
print(weights_hidden2_output)

