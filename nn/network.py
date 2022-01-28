import matplotlib.pyplot as plt
import numpy as np

from nn.layer import Layer, HiddenLayer
from nn.loss import Loss, CrossEntropy
from nn.regularization import Regularization


class Network:
    """Class used to create, train, and use a neural network.

    This class allows generation of a whole neural network in a few lines of code.
    This is the highest level of abstraction for the nn package implemented here.
    """

    def __init__(self,
                 loss_function: Loss = CrossEntropy,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 wreg: float = 0.01,
                 wrt: Regularization = None,
                 verbose: bool = False,
                 ):
        """
        :param loss_function: The loss function used at the output of the network.
        :param learning_rate: The global learning rate. Single layers can override this locally.
        :param batch_size: How many samples are passed through the network before updating parameters in the layers.
        :param wreg: The weight regularization.
        :param wrt: The weight regularization type.
        """

        self.layers: list[Layer] = []
        self.loss_function: Loss = loss_function()
        self.learning_rate: int = learning_rate
        self.batch_size: int = batch_size
        self.wreg: float = wreg
        self.wrt: Regularization = wrt()

        self.verbose = verbose

        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []


    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Propagates a single input sample through the whole network (input-, hidden-, and output layers).

        :param X: The input tensor (a single sample).
        :return: The output tensor (the predicted value based on the input).
        """

        output = X
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output

    def _backward_pass(self, J_L_S: np.ndarray) -> None:
        """Propagates the computed loss gradient through the whole network.

        :param J_L_S: The gradient with respect to output.
        """

        J_L_N = J_L_S
        for layer in self.layers[::-1]:
            J_L_N = layer.backward_pass(J_L_N)

    def _update_parameters(self) -> None:
        """Updates the parameters in all hidden layers (only layers with parameters) of the network."""

        layer: HiddenLayer
        for layer in [layer for layer in self.layers if type(layer) == HiddenLayer]:
            # if layer.size == 5:
            #     print("W", layer.W, '\ndW', np.mean(layer.W_gradients, axis=0))
            layer.update_parameters()


    def add_layer(self, layer: Layer) -> None:
        """Appends a layer to the network.

        The layer is appended to a list of layers. Be sure to add layers in the correct order, with correct
        input/output dimensions.

        :param layer: The layer to be appended to the network.
        """

        self.layers.append(layer)

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            epochs: int = 1
            ) -> None:
        """Fits the parameters of the network to the training data.

        After fitting/training, loss and accuracy can be visualized using
        visualize_loss() and visualize_accuracy() respectively.

        :param X_train: The training data.
        :param y_train: The labels corresponding to the training data
        :param epochs: Number of times the whole training set is passed through the network.
        """

        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

        print('Fitting model to data...')
        for epoch in range(epochs):
            print(f'Epoch {epoch}')

            # Train
            aggregated_train_loss: int = 0
            num_train_correct: int = 0
            for i, (X, y) in enumerate(zip(X_train, y_train)):
                y_hat = self._forward_pass(X)
                aggregated_train_loss += self.loss_function(y_hat, y)

                # Create prediction and check if correct
                prediction = np.zeros(y_hat.shape)
                prediction[np.argmax(y_hat)] = 1
                if np.array_equal(prediction, y):
                    num_train_correct += 1

                # Perform backprop by propagating the jacobian of the loss with respect to the (softmax) output
                # through the network.
                J_L_S = self.loss_function.gradient(y_hat, y)
                self._backward_pass(J_L_S)


                # If batch size has been processed, update weights
                if (epoch*len(X_train) + i + 1) % self.batch_size == 0:
                    self._update_parameters()

            # Record train loss and accuracy
            train_loss = aggregated_train_loss / (i + 1)
            train_accuracy = num_train_correct / (i + 1)
            self.train_loss.append([epoch, train_loss])
            self.train_accuracy.append([epoch, train_accuracy])

            if self.verbose:
                print('Train Loss: ', train_loss)
                print('Train Accuracy: ', train_accuracy)

            # Validation
            if X_val is not None and y_val is not None:
                aggregated_val_loss: int = 0
                num_val_correct: int = 0
                for i, (X, y) in enumerate(zip(X_val, y_val)):
                    y_hat = self._forward_pass(X)
                    aggregated_val_loss += self.loss_function(y_hat, y)

                    # Create prediction and check if correct
                    prediction = np.zeros(y_hat.shape)
                    prediction[np.argmax(y_hat)] = 1
                    if np.array_equal(prediction, y):
                        num_val_correct += 1

                # Record val loss and accuracy
                val_loss = aggregated_val_loss / (i + 1)
                val_accuracy = num_val_correct / (i + 1)
                self.val_loss.append([epoch, val_loss])
                self.val_accuracy.append([epoch, val_accuracy])

                if self.verbose:
                    print('Validation Loss: ', val_loss)
                    print('Validation Accuracy: ', val_accuracy)


        # Convert to numpy arrays
        self.train_loss = np.array(self.train_loss)
        self.val_loss = np.array(self.val_loss)
        self.train_accuracy = np.array(self.train_accuracy)
        self.val_accuracy = np.array(self.val_accuracy)

    def visualize_loss(self) -> None:
        """Visualizes training and validation loss recorded during the previous fit of the network."""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot train loss
        train_x = self.train_loss[:, 0]
        train_y = self.train_loss[:, 1]
        ax.plot(train_x, train_y, label='Train loss')

        # Plot val loss if it has been recorded
        if self.val_loss.shape[0] > 0:
            val_x = self.val_loss[:, 0]
            val_y = self.val_loss[:, 1]
            ax.plot(val_x, val_y, label='Validation loss')

        ax.legend()

        ax.set_title(f'{self.loss_function} loss during training')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')

        plt.show()

    def visualize_accuracy(self) -> None:
        """Visualizes training and validation accuracy recorded during the previous fit of the network."""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot train loss
        train_x = self.train_accuracy[:, 0]
        train_y = self.train_accuracy[:, 1]
        ax.plot(train_x, train_y, label='Train Accuracy')

        # Plot val loss if it has been recorded
        if self.val_accuracy.shape[0] > 0:
            val_x = self.val_accuracy[:, 0]
            val_y = self.val_accuracy[:, 1]
            ax.plot(val_x, val_y, label='Validation Accuracy')

        ax.legend()

        ax.set_title(f'Accuracy during training')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy[\%]')

        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the output label of a given input sample.

        :param X: The input tensor (sample).
        :return: The predicted output label.
        """

        y_hat = self._forward_pass(X)
        prediction = np.zeros(y_hat.shape)
        prediction[np.argmax(y_hat)] = 1
        return prediction

    def __str__(self):
        """Prints all layers and hyperparameters of the network."""

        out = '--------Layers--------\n'
        for layer in self.layers:
            out += layer.__str__() + '\n'
        
        out += '---Hyperparameters---\n'
        out += f'Loss function: \t {self.loss_function}\n'
        out += f'Learning rate: \t {self.learning_rate}\n'

        return out
