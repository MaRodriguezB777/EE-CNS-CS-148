import numpy as np
import sklearn

# TODO: Consider Loss Class
def negative_log_likelihood(predicted, actual):
    samples = len(actual)
    correct_logprobs = -np.log(predicted[range(samples), actual])
    data_loss = np.sum(correct_logprobs) / samples
    return data_loss

def nll_derivative(predicted, actual):
    num_samples = len(actual)
    ## compute the gradient on predictions
    dscores = predicted
    dscores[range(num_samples), actual] -= 1
    dscores /= num_samples
    return dscores

def cross_entropy(predicted, actual):
    """Given model outputs (logits) and the indexes of the true class label, computes the softmax cross entropy"""
    true_class_logits = predicted[np.arange(len(predicted)), actual]
    cross_entropy = - true_class_logits + np.log(np.sum(np.exp(predicted), axis=-1))
    return np.mean(cross_entropy)

def cross_entropy_derivative(predicted, actual):
    ones_true_class = np.zeros_like(predicted)
    ones_true_class[np.arange(len(predicted)), actual] = 1
    softmax = np.exp(predicted) / np.exp(predicted).sum(axis=-1, keepdims=True)
    return (-ones_true_class + softmax) / predicted.shape[0]


def hinge(predicted, actual):
    # TODO Part 5
    n_samples, n_classes = predicted.shape
    loss = np.zeros(n_samples)
    for i in range(n_samples):
        for j in range(n_classes):
            if j == actual[i]:
                continue
            margin = predicted[i, j] - predicted[i, actual[i]] + 1
            loss[i] += max(0, margin)
    return loss.mean()

def hinge_derivative(predicted, actual):
    # TODO Part 5
    n_samples, n_classes = predicted.shape
    derivative = np.zeros_like(predicted)
    for i in range(n_samples):
        for j in range(n_classes):
            z = actual[i] * predicted[i, j]
            if z < 1:
                derivative[i, j] = -actual[i]
            else:
                derivative[i, j] = 0
    return derivative

def mse(predicted, actual):
    # TODO Part 5
    # Hint: Convert actual to one-hot encoding
    actual_one_hot = np.zeros(predicted.shape)
    actual_one_hot[np.arange(len(actual)), actual] = 1
    return np.mean((actual_one_hot - predicted)**2)

def mse_derivative(predicted, actual):
    # TODO Part 5
    actual_one_hot = np.eye(predicted.shape[1])[actual]
    return 2 * (predicted - actual_one_hot) / len(actual)
