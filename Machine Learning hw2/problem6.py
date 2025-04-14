import numpy as np

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Derivative of Sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Binary cross-entropy loss
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-13
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# XOR dataset (which is not linearly-separable, I discussed it in Problem 1)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])  # Labels

# Initialize weights and biases
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 11000

# Weights and biases
# First, we fill them with random numbers
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training
for epoch in range(epochs):
    # Forward pass
    # I separately calculated z1 and z2 to not get confused in operations
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Computing loss
    loss = binary_cross_entropy(y, a2)

    # Backpropagation
    error_output = a2 - y
    dW2 = np.dot(a1.T, error_output * sigmoid_derivative(a2))
    db2 = np.sum(error_output * sigmoid_derivative(a2), axis=0, keepdims=True)

    error_hidden = np.dot(error_output * sigmoid_derivative(a2), W2.T)
    dW1 = np.dot(X.T, error_hidden * sigmoid_derivative(a1))
    db1 = np.sum(error_hidden * sigmoid_derivative(a1), axis=0, keepdims=True)

    # Updating weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# Final output
print("\nFinal predictions:")
print(a2.round())
