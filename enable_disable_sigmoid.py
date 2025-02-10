import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def backward_propagation(X, y, weights_input_hidden, weights_hidden_output, hidden_layer_output, output, learning_rate, enable_sigmoid=True):
    if enable_sigmoid:
        output_delta = (y - output) * sigmoid_derivative(output)
        hidden_delta = sigmoid_derivative(hidden_layer_output) * (output_delta.dot(weights_hidden_output.T))
    else:
        output_delta = (y - output)
        hidden_delta = output_delta.dot(weights_hidden_output.T)

    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

    return weights_input_hidden, weights_hidden_output

def forward_propagation(X, weights_input_hidden, weights_hidden_output, enable_sigmoid=True):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    if enable_sigmoid:
        hidden_layer_output = sigmoid(hidden_layer_input)
    else:
        hidden_layer_output = hidden_layer_input
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    if enable_sigmoid:
        output = sigmoid(output_layer_input)
    else:
        output = output_layer_input

    return hidden_layer_output, output


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.5


weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)


enable_sigmoid = True


hidden_layer_output, output = forward_propagation(X, weights_input_hidden, weights_hidden_output, enable_sigmoid)


weights_input_hidden, weights_hidden_output = backward_propagation(
    X, y, weights_input_hidden, weights_hidden_output, hidden_layer_output, output, learning_rate, enable_sigmoid
)


print("Weight: Input to hidden")
print(weights_input_hidden)
print("Weight: hidden to output")
print(weights_hidden_output)
