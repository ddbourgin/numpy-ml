import numpy as np
import pandas as pd


class GaussianDiscriminantAnalysis():  ## requires data as a pandas dataframe in the format [atribute, atribute, ....., class]
    def __init__(self, df):
        self.data = df
        self.N = self.data.shape[0]
        self.classes = list(set(self.data.iloc[:, -1]))
        self.splits = []
        self.means = []
        self.covariances = []
        for label in self.classes:
            split = self.data.loc[self.data.iloc[:, -1] == label].iloc[:, 0:-1].astype(float)
            self.splits.append(split)
            self.means.append((np.mean(np.asarray(split),axis=0)))
            self.covariances.append(np.cov(np.asarray(split), rowvar=False))

    def gaussian_Probability(self, x, mean, covariance):
        n = np.shape(x)[0]
        nominator = np.exp((-0.5) * (np.dot(np.transpose(x - mean), np.matmul(np.linalg.inv(covariance), (x - mean)))))
        denominator = (((2.0 * np.pi) ** (n / 2.0)) * np.sqrt(np.linalg.norm(covariance)))
        return np.divide(nominator, denominator)

    def classify(self, datapoint):
        # print("-------------------\n")
        # print(datapoint)
        classification = {}
        p_of_x_given_y = []
        P_of_Y = []
        P_of_Y_given_X = []
        for index, label in enumerate(self.classes):
            split = self.data.loc[self.data.iloc[:, -1] == label].iloc[:, 0:-1]
            # print(index,label)
            P_of_Y.append(len(split) / (self.N))
            p_of_x_given_y.append(self.gaussian_Probability(datapoint, self.means[index], self.covariances[index]))
        P_of_X = sum([p_of_x_given_y[i] * P_of_Y[i] for i in range(len(self.classes))])
        # print(P_of_Y,"P(Y)")
        # print(p_of_x_given_y,"P(X|Y)")
        # print(P_of_X, "P(X)")#
        for index, label in enumerate(self.classes):
            P_of_Y_given_X.append((p_of_x_given_y[index] * P_of_Y[index]) / P_of_X)
            classification.update({str(label): (p_of_x_given_y[index] * P_of_Y[index]) / P_of_X})

        return classification
