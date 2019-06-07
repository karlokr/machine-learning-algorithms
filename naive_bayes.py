import numpy as np


class NaiveBayesClassifier():
    def __init__(self):
        self.classes = None
        self.priors = None
        self.u = None
        self.sigma = None

    def __classPriors(self, y):
        priors = np.zeros(np.shape(self.classes)[0])
        n = np.shape(y)[0]
        for i in self.classes:
            sum_y = 0
            for y_i in y:
                if y_i == i:
                    sum_y = sum_y + 1
            priors[int(i)-1] = sum_y/n
        return priors

    def __classConditionalDensity(self, X, y):
        u = []
        sigma = []
        for i in self.classes:
            # total number of occurences of class y
            n_y = 0
            for y_i in y:
                if y_i == i:
                    n_y = n_y + 1

            # u_i's
            sum_x = 0
            for k in range(len(y)):
                if y[k] == i:
                    sum_x = sum_x + X[k]
            u_i = np.matrix(sum_x / n_y)
            u.append(u_i)

            # sigma_i's
            sum_sigma = np.matrix(
                np.zeros((np.shape(X[0])[0], np.shape(X[0])[0])))
            for k in range(len(y)):
                if y[k] == i:
                    xk = np.matrix(X[k]) - u[int(i-1)]
                    sum_sigma = sum_sigma + np.outer(xk, xk)
            sigma_i = sum_sigma / n_y
            sigma.append(sigma_i)

        return u, sigma

    def train(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.priors = self.__classPriors(y_train)
        self.u, self.sigma = self.__classConditionalDensity(X_train, y_train)

    def predict(self, X_test):
        classProbs = []
        predictions = []

        for x in X_test:
            prob = []
            for i in self.classes:
                I = int(i-1)
                sigma_inv = np.linalg.inv(self.sigma[I])
                sigma_det = np.linalg.det(self.sigma[I])
                e = -0.5 * (np.matrix(x) - self.u[I]) * \
                    sigma_inv * (np.matrix(x) - self.u[I]).T
                exp = np.exp(e[0, 0])
                predict = self.priors[I] * np.power(sigma_det, -0.5) * exp
                prob.append(predict)

            normed = prob/np.linalg.norm(prob, ord=1)
            classProbs.append(normed)

        for i in range(len(classProbs)):
            predictions.append(classProbs[i].argmax(axis=0) + 1)

        return predictions, classProbs


###############################
# Example Usage
###############################
# import training data
X_train = np.genfromtxt('X_train.csv', delimiter=",")
y_train = np.genfromtxt('y_train.csv')
# import test data
X_test = np.genfromtxt('X_test.csv', delimiter=",")

clf = NaiveBayesClassifier()  # initialize the classifier
clf.train(X_train, y_train)  # train the classifier using the training data
predictions, probabilities = clf.predict(
    X_test)  # make predictions on new cases
