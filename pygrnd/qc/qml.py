'''Copyright 2022 JoS QUANTUM GmbH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''



import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt



def mse_loss(y_real: np.array, y_pred: np.array) -> float:
    """Returns the mean squared error between the model predictions and the real value.

    Args:
        y_real (np.array):
            actual values.
        y_pred (np.array):
            model predictions.
    """
    assert len(y_real)==len(y_pred)
    loss = 0
    for y1, y2 in zip(y_real, y_pred):
        loss += (y1 - y2)**2
    loss /= len(y_real)
    return loss



def accuracy_loss(y_real: np.array, y_pred: np.array, threshold: float=0.5) -> float:
    """Returns the accuracy loss between model predictions and actual values.

    Args:
        y_real (np.array):
            actual values.
        y_pred (np.array):
            model predictions.
        threshold (float, optional):
            Prediction threshold. Defaults to 0.5.
    """
    assert len(y_real)==len(y_pred)
    y_ = [int(y >= threshold) for y in y_pred]
    loss = 0
    for y1, y2 in zip(y_real, y_pred):
        if y1 == y2:
            loss += 1
    loss /= len(y_real)
    return loss



def bce_loss(y_real: np.array, y_pred: np.array, eps:float=1e-8) -> float:
    """Returns the binary cross entropy loss between model predictions and actual values.

    Args:
        y_real (np.array):
            actual values.
        y_pred (np.array):
            model predictions.
        eps (_type_, optional):
            small constant to avoid computing log(0). Defaults to 1e-8.
    """
    assert len(y_real)==len(y_pred)

    loss = 0
    for y1, y2 in zip(y_real, y_pred):
        loss += y1*np.log(y2 + np.abs(eps*np.random.randn())) + (1 - y1)*np.log(1 - y2 + np.abs(eps*np.random.randn()))

    loss *= -1/len(y_real)

    return loss





def one_epoch(qnode: qml.QNode, weights: np.tensor, loss_function: function, opt, batch_size: int,
              x_train: np.array, y_train: np.array):
    """Trains the given parametrized quantum circuit as a quantum neural network accross the given
    data. Returns the updated optimized parameters for this learning epoch.

    Args:
        qnode (_type_):
            Input PQC.
        weights (_type_):
            Parameters of the PQC.
        opt (_type_):
            Optimizer that will classically optimize the cost function of the model
        batch_size (_type_):
            batch size.
        x_train (_type_):
            Train data input.
        y_train (_type_):
            Train data output.

    """
    N = len(y_train)
    indices = list(range(N))
    np.random.shuffle(indices)

    for batch in range(N//batch_size):
        x_batch = x_train[indices[batch*batch_size:(batch+1)*batch_size]]
        y_batch = y_train[indices[batch*batch_size:(batch+1)*batch_size]]

        def cost(weights, qnode, loss_function, x_batch, y_batch):
            y_pred = [(qnode(weights, x) + 1)/2 for x in x_batch]
            return loss_function(y_batch, y_pred)

        cost_function = lambda w: cost(w, qnode=qnode, loss_function=loss_function, x_batch=x_batch, y_batch=y_batch)
        
        weights = opt.step(cost_function, weights)

    return weights





class QNNClassifier():
    """This class is used to employ a given parametrized quantum circuit (PQC) as a quantum neural
    network for a binary classification task (output must be 0 and 1).

    Attributes
    ----------
    qnode :
        Pennylane parametrized quantum circuit used as a trainable model (QNN).
    weights_shape :
        Shape of the vector of parameters used for the quantum circuit.
    loss function :
        Loss function that will be used to compare the model output and the real data in order
        to classically optimize the parameters of the model. Defaults to Binary Cross Entropy.

    Methdods
    --------
    fit(x_train, y_train, x_test, y_test, epochs, batch_size, optimizer, learning_rate, verbose):
        Fits the model by performing supervised learning on the set (x_train, y_train).
    predict_probas(x_list):
        Returns the vector of probabilities (model output) for the given input list. The
        probability corresponds to the model confidence in predicting 1 as output.
    predict(x_list, threshold):
        Return the predictions of the model (probabilities above the given threshold are
        classified as 1 and the rest as 0).
    """



    def __init__(self, qnode: qml.Qnode, weights_shape:tuple, loss_function: function = bce_loss):
        """Constructs the classification model

        Args:
            qnode (_type_):
                Pennylane parametrized quantum circuit used as a trainable model (QNN).
            weights_shape (_type_):
                Shape of the vector of parameters used for the quantum circuit.
            loss_function (_type_, optional): 
                Loss function that will be used to compare the model output and the real data in order
                to classically optimize the parameters of the model. Defaults to Binary Cross Entropy.
        """
        self.qnode = qnode
        n_layers, n_qubits, k = weights_shape
        np.random.seed(42)
        self.weights = np.random.randn(n_layers, n_qubits, k, requires_grad=True)
        self.loss_function = loss_function



    def fit(self, x_train: np.array, y_train: np.array, epochs: int, batch_size: int, optimizer,
            learning_rate: float, threshold: float=0.5, verbose: bool=True):
        """Fits the model by performing supervised learning on the set (x_train, y_train). 
        A classical optimizer is used to minimize a cost function (self.loss_function evaluated
        between the model predictions and real data). During each training epoch, the model goes
        through the whole training set (x_train, y_train). The training set is divided into
        training batches. Gradients are computed and weights updates at the end of each batch.

        Args:
            x_train (np.array):
                Training inputs.
            y_train (np.array):
                Training outputs. Must be elements of {0, 1}.
            epochs (int):
                Number of training epochs.
            batch_size (int):
                Batch size (number of samples after which the weights are updated).
            optimizer (_type_):
                Classical optimizer used to minimize the loss function.
            learning_rate (float):
                Learning rate of the classical optimizer.
            threshold (float, optional): 
                Prediction threshold. The model outputs probabilites between 0 and 1, a probability
                above the threshold will be classified as 1 and the others as 0. Defaults to 0.5.
            verbose (bool, optional):
                Whether to print the losses evolution at each training epoch. Defaults to True.

        Returns:
            train_losses:
                List of train losses at each training epoch.
        """
        opt = optimizer(learning_rate)

        train_losses = []
        
        for epoch in range(epochs):
            self.weights = one_epoch(qnode=self.qnode, weights=self.weights, 
                                     loss_function=self.loss_function, opt=opt,
                                     batch_size=batch_size, x_train=x_train, y_train=y_train)

            pred_epoch = self.predict_probas(x_train)
            train_loss = self.loss_function(y_train, pred_epoch)
            train_acc = accuracy_loss(y_train, [int(x>=threshold) for x in pred_epoch])

            train_losses.append(train_loss)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs},   train loss = {train_loss},   train accuracy = {train_acc}')

        return train_losses


    def predict_probas(self, x_list: np.array):
        """Returns the vector of probabilities (model output) for the given input data. The
        probability corresponds to the model confidence in predicting 1 as output.

        Args:
            x_list (np.array):
                Input data.
        """
        return [(self.qnode(self.weights, x) + 1)/2 for x in x_list]



    def predict(self, x_list: np.array, threshold: float=0.5):
        """Return the predictions of the model. The model outputs probabilites between 0 and 1,
        a probability above the threshold will be classified as 1 and the rest as 0.

        Args:
            x_list (np.array):
                Input data.
            threshold (float, optional):
                Prediction threshold. . Defaults to 0.5.
        """

        probas = self.predict_probas(x_list)
        return [int(x>=threshold) for x in probas]





class QNNRegressor():
    """This class is used to employ a given parametrized quantum circuit (PQC) as a quantum neural
    network for a regression task.

    Attributes
    ----------
    qnode :
        Pennylane parametrized quantum circuit used as a trainable model (QNN).
    weights_shape :
        Shape of the vector of parameters for the quantum circuit.
    loss function :
        Loss function that will be used to compare the model output and the real data in order
        to classically optimize the parameters of the model. Defaults to MSE loss.

    Methdods
    --------
    fit(x_train, y_train, x_test, y_test, epochs, batch_size, optimizer, learning_rate, verbose):
        Fits the model by performing supervised learning on the set (x_train, y_train).
    predict(x_list):
        Returns the predictions (model outputs) for the given input data.
    """



    def __init__(self, qnode, weights_shape, loss_function=mse_loss):
        """Constructs the regression model

        Args:
            qnode (_type_):
                Pennylane parametrized quantum circuit used as a trainable model (QNN).
            weights_shape (_type_):
                Shape of the vector of parameters used for the quantum circuit.
            loss_function (_type_, optional): 
                Loss function that will be used to compare the model output and the real data in order
                to classically optimize the parameters of the model. Defaults to Binary Cross Entropy.
        """
        self.qnode = qnode
        n_layers, n_qubits, k = weights_shape
        np.random.seed(42)
        self.weights = np.random.randn(n_layers, n_qubits, k, requires_grad=True)
        self.loss_function = loss_function



    def fit(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array,
            epochs: int, batch_size: int, optimizer, learning_rate:float, verbose: bool=True):
        """Fits the model by performing supervised learning on the set (x_train, y_train). 
        A classical optimizer is used to minimize a cost function (self.loss_function evaluated
        between the model predictions and real data). During each training epoch, the model goes
        through the whole training set (x_train, y_train). The training set is divided into
        training batches. Gradients are computed and weights updates at the end of each batch.

        Args:
            x_train (np.array):
                Training inputs.
            y_train (np.array):
                Training outputs. Must be elements of [0, 1].
            x_test (np.array):
                Testing inputs.
            y_test (np.array):
                Testing outputs. Must be elements of [0, 1].
            epochs (int):
                Number of training epochs.
            batch_size (int):
                Batch size (number of samples after which the weights are updated).
            optimizer (_type_):
                Classical optimizer used to minimize the loss function.
            learning_rate (float):
                Learning rate of the classical optimizer.
            verbose (bool, optional):
                Whether to print the losses evolution at each training epoch. Defaults to True.

        Returns:
            train_losses:
                List of train losses at each training epoch.
            test_losses:
                List of test losses at each training epoch.
        """
        opt = optimizer(learning_rate)

        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            self.weights = one_epoch(qnode=self.qnode, weights=self.weights,
                                     loss_function=self.loss_function, opt=opt, 
                                     batch_size=batch_size, x_train=x_train, y_train=y_train)

            pred_epoch = self.predict(x_train)
            train_loss = self.loss_function(y_train, pred_epoch)
            train_losses.append(train_loss)

            y_hat = self.predict(x_test)
            test_loss = self.loss_function(y_test, y_hat)
            test_losses.append(test_loss)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs},   train loss = {train_loss} test_loss = {test_loss}')

        return train_losses, test_losses
    

    def predict(self, x_list: np.array):
        """Returns the predictions (model outputs) for the given input data.

        Args:
            x_list (np.array):
                Input data.
        """
        return [(self.qnode(self.weights, x) + 1)/2 for x in x_list]