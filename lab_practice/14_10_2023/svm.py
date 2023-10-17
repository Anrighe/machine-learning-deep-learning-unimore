import numpy as np
from numpy.linalg import norm

eps = np.finfo(float).eps


class SVM:
    """ Models a Support Vector machine classifier based on the PEGASOS algorithm. """

    def __init__(self, n_epochs, lambDa, use_bias=True):
        """ Constructor method """

        # weights placeholder
        self._w = None
        self._original_labels = None
        self._n_epochs = n_epochs
        self._lambda = lambDa
        self._use_bias = use_bias

    def map_y_to_minus_one_plus_one(self, y):
        """
        Map binary class labels y to -1 and 1
        """
        ynew = np.array(y)
        self._original_labels = np.unique(ynew)
        assert len(self._original_labels) == 2
        ynew[ynew == self._original_labels[0]] = -1.0
        ynew[ynew == self._original_labels[1]] = 1.0
        return ynew

    def map_y_to_original_values(self, y):
        """
        Map binary class labels, in terms of -1 and 1, to the original label set.
        """
        ynew = np.array(y)
        ynew[ynew == -1.0] = self._original_labels[0]
        ynew[ynew == 1.0] = self._original_labels[1]
        return ynew

    def loss(self, y_true, y_pred):
        """
        The PEGASOS loss term

        Parameters
        ----------
        # ground-truth labels array y_true along with its corresponding model predictions y_pred as inputs
        y_true: np.array
            real labels in {0, 1}. shape=(n_examples,)
        y_pred: np.array
            predicted labels in [0, 1]. shape=(n_examples,)

        Returns
        -------
        float
            the value of the pegasos loss.
        """
        reg_loss = 0.5 * self._lambda * norm(self._w)
        l_h = np.maximum(np.zeros((y_true.shape[0],)), 1.0 - y_true * y_pred)
        hinge_loss = np.mean(l_h)

        return reg_loss + hinge_loss

    def fit_gd(self, X, Y, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        verbose: bool
            whether or not to print the value of cost function.
        """

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0],1),dtype=X.dtype)], axis=-1)

        n_samples, n_features = X.shape
        Y = self.map_y_to_minus_one_plus_one(Y)

        # initialize weights
        self._w = np.zeros(shape=(n_features,), dtype=X.dtype)

        t = 0
        # loop over epochs
        for e in range(1, self._n_epochs+1):
            for j in range(n_samples):
                X_j, y_j = X[j,:], Y[j]
                t += 1
                n_t = 1.0/(t*self._lambda)
                X_j_pred = np.dot(X_j, self._w)
                if y_j * X_j_pred < 1:
                    self._w = (1.0 - n_t*self._lambda)*self._w + n_t*y_j*X_j
                else:
                    self._w = (1.0 - n_t*self._lambda)*self._w

            # predict training data
            cur_prediction = np.dot(X, self._w)

            # compute (and print) cost
            cur_loss = self.loss(y_true=Y, y_pred=cur_prediction)

            if verbose:
                print("Epoch {} Loss {}".format(e, cur_loss))

    def predict(self, X):

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0],1),dtype=X.dtype)], axis=-1)

        """
        Write HERE the criterium used during inference. 
        W * X > 0 -> positive class
        X * X < 0 -> negative class
        """

        # returns a random value using the np.where(cond, a, b) function, 
        # which takes 3 numpy.array with the same shape (or broadcastable to the
        # same shape) as input. The function returns a new numpy.array, containing:
        # 1. the corresponding element of a for each position where the boolean array
        #   cond contains a True value;
        # 2. the corresponding element of b for each position where the boolean array
        #   cond contains a False value.

        return np.where(np.dot(X, self._w) > 0.0, self._original_labels[1], self._original_labels[0])

