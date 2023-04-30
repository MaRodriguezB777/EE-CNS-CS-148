import numpy as np
from datasets import *
from random_control import *
from losses import *


class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

    def softmax(self, inputs):
        # (DONE) (Part 2)
        exp = np.exp(inputs)
        sum = np.sum(exp, axis=1)
        result = np.array([exp[i,:]/sum[i] for i in range(exp.shape[0])])
        return result

    def tanH(self, inputs):
        return np.tanh(inputs)


    def sigmoid(self, inputs):
        # (DONE) (Part 2)
        return np.divide(np.array([1]), (1 + np.exp( - inputs)))

    def relu(self, inputs):
        # (DONE) (Part 5)
        return np.maximum(np.zeros_like(inputs), inputs)

    def tanH_derivative(self, Z):
        # (DONE) (Part 5)
        return 1 - self.tanH(Z)**2

    def sigmoid_derivative(self, Z):
        # (DONE) (Part 2)
        sig = self.sigmoid(Z)
        return sig * (1 - sig)

    def relu_derivative(self, Z):
        # (DONE) (Part 5)
        return Z > 0

    def apply_chain_rule_activation_derivative(self, q, activation_derivative):
        # TODO rename the variable q appropriately -- what should this be?
        return None

    def forward(self, inputs, weights, bias, activation):
        '''
        inputs = (N x C) where N is the number of inputs and C is the size of the embedding of each input
        weights = (O x C) where O is the dimension of the outputted layer. There are O neurons in the outputted layer
        bias = (1 x O) bias term of the weight for each output

        inputs * weights^T = each row is the resulting embedding of each output neuron based on taking a a weighted sum of each of the
        inputs, weighing based on the different embeddings (the size of each input).

        '''
        # weights.shape = (prev_layer, curr_layer)

        # Each Z_curr's row is treated as the z for the ith neuron.
        # inputs = np.concatenate((np.ones(shape=(inputs.shape[0],1)) ,inputs), axis=1) # Add x_0 = 1 to each row
        # weights = np.concatenate((bias, weights), axis=1) # Add w_0 = b_i to each row where b_i is the bias for output i

        Z_curr = np.add(np.matmul(inputs, weights.T), bias)  # (DONE) TODO compute Z_curr from weights and bias
        
        if activation == 'relu':
            A_curr = self.relu(inputs=Z_curr)
        elif activation == 'sigmoid':
            A_curr = self.sigmoid(inputs=Z_curr)
        elif activation == 'tanH':
            A_curr = self.tanH(inputs=Z_curr)
        elif activation == 'softmax':
            A_curr = self.softmax(inputs=Z_curr)
        else:
            raise ValueError('Activation function not supported: ' + activation)

        return A_curr, Z_curr

    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):

        # (DONE) TODO each of these functions require you to compute all of the colored terms in the Part 2 Figure.
        # We will denote the partial derivative of the loss with respect to each variable as dZ, dW, db, dA
        # These variable map to the corresponding terms in the figure. Note that these are matrices and not individual
        # values, you will determine how to vectorize the code yourself. Think carefully about dimensions!
        # You can use the self.apply_chain_rule_activation_derivative() function, although there are solutions without it.
        '''
        The inputs to this function are:
            dA_curr - (N x O) the partial derivative of the loss with respect to the activation of the preceding layer (l + 1)
                        for each input.
            W_curr - (O x I) the weights of the layer (l). Corresponds to the contributions of each component of the input C to each
                        output neuron.
            Z_curr - (N x O) the weighted sum of layer (l). Corresponds to the signal from the current layer to the activation function
                        of the next layer for each input.
            A_prev - (N x I) the activation of this layer (l) ... we use prev with respect to dA_curr. Corresponds to the neurons of
                        this layer (outputs of activation function from last layer) for each input.

        The outputs are the partial derivatives with respect
            dA - the activation of this layer (l) -- needed to continue the backprop
            dW - (O x I) the weights -- needed to update the weights
            db - the bias -- (needed to update the bias
        '''

        # print("dA_curr = ", dA_curr.shape) # (N x O) Gives the change in activation function for jth neuron for each ith input.
        # print("W_curr = ", W_curr.shape) # (O x I) For each output, the contribution of each neuron in layer l
        # print("Z_curr = ", Z_curr.shape) # (N x O) Gives the signal corresponding to the Nth input on the Oth output.
        # print("A_prev = ", A_prev.shape) # (N x I) 

        # A_prev^T. Each row gives the ith neuron at the layer for each jth input.
        
        # dW = A_prev^T*dA_curr*activation_derivative = (I x O). Gives the change in the weights from layer to next layer, summed over all
        #   the inputs x which is why there is no N term in output dimensions.

        # dA = dZ*W_curr = (N x I). Gives the change over all outputs (and distinguishing between each input) of each of the input neurons
        #   in this layer
        if activation == 'softmax':
            # We deal with the softmax function for you, so dZ is not needed for this one. dA_curr = dZ for this one.
            dZ = dA_curr
            dW = np.matmul(dZ.T, A_prev)
            db = np.sum(dZ, axis=0)
            dA = np.matmul(dZ, W_curr)
        elif activation == 'sigmoid':
            # Computing dZ is not technically needed, but it can be used to help compute the other values.
            activation_derivative = self.sigmoid_derivative(Z_curr)
            dZ = np.multiply(dA_curr, activation_derivative)
            dW = np.matmul(dZ.T, A_prev)
            db = np.sum(dZ, axis=0)
            dA = np.matmul(dZ, W_curr)
        elif activation == 'tanH':
            activation_derivative = self.tanH_derivative(Z_curr)
            dZ = np.multiply(dA_curr, activation_derivative)
            dW = np.matmul(dZ.T, A_prev)
            db = np.sum(dZ, axis=0)
            dA = np.matmul(dZ, W_curr)
        elif activation == 'relu':
            activation_derivative = self.relu_derivative(Z_curr)
            dZ = np.multiply(dA_curr, activation_derivative)
            dW = np.matmul(dZ.T, A_prev)
            db = np.sum(dZ, axis=0)
            dA = np.matmul(dZ, W_curr)
        else:
            raise ValueError('Activation function not supported: ' + activation)

        return dA, dW.T, db # Need to do dW.T for tests to pass

'''
* `MLP` is a class that represents the multi-layer perceptron with a variable number of hidden layer. 
   The constructor initializes the weights and biases for the hidden and output layers.
* `sigmoid`, `relu`, `tanh`, and `softmax` are activation function used in the MLP. 
   They should each map any real value to a value between 0 and 1.
* `forward` computes the forward pass of the MLP. 
   It takes an input X and returns the output of the MLP.
* `sigmoid_derivative`, `relu_derivative`, `tanH_derivative` are the derivatives of the activation functions. 
   They are used in the backpropagation algorithm to compute the gradients.
*  `mse_loss`, `hinge_loss`, `cross_entropy_loss` are each loss functions.
   The MLP algorithms optimizes to minimize those.
* `backward` computes the backward pass of the MLP. It takes the input X, the true labels y, 
   the predicted labels y_hat, and the learning rate as inputs. 
   It computes the gradients and updates the weights and biases of the MLP.
* `train` trains the MLP on the input X and true labels y. It takes the number of epochs 
'''

class MLP:
    def __init__(self, layer_list):
        '''
        Arguments
        --------------------------------------------------------
        layer_list: a list of numbers that specify the width of the hidden layers. 
               The dataset dimensionality (input layer) and output layer (1) 
               should not be specified.
        '''
        self.layer_list = layer_list
        self.network = []  ## layers
        self.architecture = []  ## mapping input neurons --> output neurons
        self.params = []  ## W, b
        self.memory = []  ## Z, A
        self.gradients = []  ## dW, db
        self.loss = []
        self.accuracy = []

        self.loss_func = None
        self.loss_derivative = None

        self.init_from_layer_list(self.layer_list)

    # (DONE) TODO read and understand the next several functions, you will need to understand them to complete the assignment.
    #  In particular, you will need to understand
    #  self.network, self.architecture, self.params, self.memory, and self.gradients. It may be helpful to write some
    #  notes about what each of these variables are and how they are used.
    def init_from_layer_list(self, layer_list):
        for layer_size in layer_list:
            self.add(Layer(layer_size))

    def add(self, layer):
        self.network.append(layer)

    def _compile(self, data, activation_func='relu'):
        self.architecture = [] 
        for idx, layer in enumerate(self.network):
            if idx == 0:
                self.architecture.append({'input_dim': data.shape[1], 'output_dim': self.network[idx].num_neurons,
                                          'activation': activation_func})
            elif idx > 0 and idx < len(self.network) - 1:
                self.architecture.append(
                    {'input_dim': self.network[idx - 1].num_neurons, 'output_dim': self.network[idx].num_neurons,
                     'activation': activation_func})
            else:
                self.architecture.append(
                    {'input_dim': self.network[idx - 1].num_neurons, 'output_dim': self.network[idx].num_neurons,
                     'activation': 'softmax'})
        return self

    def _init_weights(self, data, activation_func, seed=None):
        self.params = []
        self._compile(data, activation_func)

        if seed is None:
            for i in range(len(self.architecture)):
                self.params.append({
                    'W': generator.uniform(low=-1, high=1,
                                           size=(self.architecture[i]['output_dim'],
                                                 self.architecture[i]['input_dim'])),
                    'b': np.zeros((1, self.architecture[i]['output_dim']))})
        else:
            # For testing purposes
            fixed_generator = np.random.default_rng(seed=seed)
            for i in range(len(self.architecture)):
                self.params.append({
                    'W': fixed_generator.uniform(low=-1, high=1,
                                           size=(self.architecture[i]['output_dim'],
                                                 self.architecture[i]['input_dim'])),
                    'b': np.zeros((1, self.architecture[i]['output_dim']))})

        

        return self

    def forward(self, data):
        A_prev = data
        A_curr = None
        
        # self.memory[0] = {'Z': data, 'A': data}

        for i in range(len(self.params)):

            # (DONE) TODO compute the forward for each layer and store the appropriate values in the memory.
            # We format our memory_list as a list of dicts, please follow this format.
            # mem_dict = {'?': ?}; self.memory.append(mem_dict)
            W_l = self.params[i]['W']
            b_l = self.params[i]['b']

            # print(W_l.shape)
            # print(b_l.shape)
            activation = self.architecture[i]['activation']

            A_curr, Z_curr = self.network[i].forward(A_prev, W_l, b_l, activation)
            self.memory.append({'Z': Z_curr, 'A': A_prev})

            A_prev = A_curr

        # print("finished forward pass! ------------------- ")
        return A_curr

    def backward(self, predicted, actual):
        ## compute the gradient on predictions
        dscores = self.loss_derivative(predicted, actual)
        dA_prev = dscores  # This is the derivative of the loss function with respect to the output of the last layer
        dA_curr = dscores

        # TODO compute the backward for each layer and store the appropriate values in the gradients.
        # We format our gradients_list as a list of dicts, please follow this format (same as self.memory).

        for i in range(len(self.network)).__reversed__():
            
            # print(str(i) + "-------------")
            # print(self.memory[i]['Z'].shape)
            # print(self.memory[i - 1]['A'].shape)
            # print(dA_curr.shape)

            dA, dW, db = self.network[i].backward(dA_prev, self.params[i]['W'], self.memory[i]['Z'], self.memory[i]['A'], self.architecture[i]['activation'])

            self.gradients.append({'dW': dW, 'db': db})
            
            # print("W and dW:")
            # print(self.params[i]['W'].shape)
            # print(self.gradients[i]['dW'].shape)
            
            dA_prev = dA
            # print(dA_curr.shape)
            # Z_curr = 

        self.gradients = self.gradients[::-1]

    def _update(self, lr):
        # TODO update the network parameters using the gradients and the learning rate.
        #  Recall gradients is a list of dicts, and params is a list of dicts, pay attention to the order of the dicts.
        #  Is gradients the same order as params? This might depend on your implementations of forward and backward.
        #  Should we add or subtract the deltas?

        # print("-----------------")
        # print(len(self.params))
        # print(len(self.gradients))
        for i in range(len(self.layer_list)):
            # gradients are direction greatest increase so we subtract gradients
            self.params[i]['W'] = self.params[i]['W'] - lr * self.gradients[i]['dW'].T
            self.params[i]['b'] = self.params[i]['b'] - lr * self.gradients[i]['db']


    # Loss and accuracy functions
    def _calculate_accuracy(self, predicted, actual):
        return np.mean(np.argmax(predicted, axis=1) == actual)

    def _calculate_loss(self, predicted, actual):
        return self.loss_func(predicted, actual)

    def _set_loss_function(self, loss_func_name='negative_log_likelihood'):
        if loss_func_name == 'negative_log_likelihood':
            self.loss_func = negative_log_likelihood
            self.loss_derivative = nll_derivative
        elif loss_func_name == 'hinge':
            self.loss_func = hinge
            self.loss_derivative = hinge_derivative
        elif loss_func_name == 'mse':
            self.loss_func = mse
            self.loss_derivative = mse_derivative
        else:
            raise Exception("Loss has not been specified. Abort")

    def get_losses(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy

    def train(self, X_train, y_train, epochs=1000, lr=1e-4, batch_size=16, activation_func='relu', loss_func='negative_log_likelihood'):

        self.loss = []
        self.accuracy = []
        self._set_loss_function(loss_func)

        # cast to int
        y_train = y_train.astype(int)

        # initialize network weights
        self._init_weights(X_train, activation_func)

        # (DONE) TODO calculate number of batches
        num_datapoints = len(y_train)
        num_batches = int(np.ceil(num_datapoints / batch_size))

        # TODO shuffle the data and iterate over mini-batches for each epoch.
        #  We are implementing mini-batch gradient descent.
        #  How you batch the data is up to you, but you should remember shuffling has to happen the same way for
        #  both X and y.

        for i in range(int(epochs)):
            
            shuffle = generator.permutation(len(X_train))
            X_train = X_train[shuffle]
            y_train = y_train[shuffle]

            batch_loss = 0
            batch_acc = 0

            for batch_num in range(int(num_batches - 1)):

                X_batch = X_train[batch_size*batch_num :
                            batch_size*(batch_num + 1)]
                y_batch = y_train[batch_size*batch_num :
                            batch_size*(batch_num + 1)]

                # (DONE) TODO Hint: do any variables need to be reset each pass?

                # Choose to clear rather than set list to None so that we do not need to add new elements to list every time
                self.gradients = []
                self.memory = []

                 # (DONE) TODO compute yhat
                yhat = self.forward(X_batch)

                # (DONE) TODO compute and update batch acc
                acc = self._calculate_accuracy(yhat, y_batch)
                # (DONE) TODO compute and update batch loss
                loss = self._calculate_loss(yhat, y_batch)
                
                batch_acc += acc
                batch_loss += loss

                # Stop training if loss is NaN, why might the loss become NaN or inf?
                if np.isnan(loss) or np.isinf(loss):
                    if(len(self.accuracy) == 0 or len(self.loss) == 0):
                        self.accuracy.append(0)
                        self.loss.append(np.inf)
                    s = 'EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}'.format(i, lr, self.accuracy[-1], self.loss[-1])
                    # print(s)
                    # print("Stopping training because loss is NaN")
                    return
                
                # Do backpropagation
                self.backward(yhat, y_batch)
                # TODO update the network
                self._update(lr)

            self.loss.append(batch_loss / num_batches)
            self.accuracy.append(batch_acc / num_batches)

            if i % 20 == 0:
                s = 'EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}'.format(i, lr, self.accuracy[-1], self.loss[-1])
                # print(s)

    def predict(self, X, y, loss_func='negative_log_likelihood'):
        # TODO predict the loss and accuracy on a val or test set and print the results. Make sure to gracefully handle
        #  the case where the loss is NaN or inf.
        
        yhat = self.forward(X)
        y = y.astype(int)
        self._set_loss_function(loss_func)

        # for plotting purposes
        self.test_loss = self._calculate_loss(yhat, y)  # (DONE) TODO loss
        self.test_accuracy = self._calculate_accuracy(yhat, y)  # (DONE) TODO accuracy


if __name__ == '__main__':
    # Copy of part2.py in case useful for debugging
    N = 100
    M = 100
    dims = 3
    gaus_dataset_points = generate_nd_dataset(N, M, kGaussian, dims).get_dataset()
    X = gaus_dataset_points[:, :-1]
    y = gaus_dataset_points[:, -1].astype(int)

    model = MLP([3,2])
    model.train(X, y)
