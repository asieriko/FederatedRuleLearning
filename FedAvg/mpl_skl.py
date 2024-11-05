# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from FedRLS.metrics import metrics

def MLP(dataset):
    X_train = dataset[0]
    X_test = dataset[1]
    y_train = dataset[2]
    y_test = dataset[3]
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return metrics(clf.predict(X_test),y_test)


class MLP_federatedC():

    def __init__(self, clients_dataset):
        self.clients_dataset = clients_dataset
        self.weights = []
        self.MLP_classifiers = [MLPClassifier(random_state=1, max_iter=300) for _ in range(len(clients_dataset))]

    def train(self):
        weights = None
        for clf, dataset in zip(self.MLP_classifiers,self.clients_dataset):
            X_train = dataset[0].to_numpy()
            y_train = dataset[2]
            clf.fit(X_train, y_train)
            # Store the coefficientes of each client
            if weights is None:
                weights = clf.coefs_
            else:
                for i in range(len(weights)):
                    weights[i] += clf.coefs_[i]
        self.weights = weights

    def avg(self):
        # Compute the average of the stored coefficients
        for i in range(len(self.weights)):
            weights[i] = self.weights[i]/len(MLP_classifiers)

    def update_weights(self):
        for clf in self.MLP_classifiers:
            for i in range(len(self.weights)):
                clf.coefs_[i] = self.weights[i]

    def eval(self):
        scores = []
        for clf, dataset in zip(self.MLP_classifiers, self.clients_dataset):
            X_test = dataset[1].to_numpy()
            y_test = dataset[3]
            scores.append(metrics(clf.predict(X_test),y_test))        
        return scores

    def fit(self):
        self.train()
        sc1 = self.eval()
        self.avg()
        self.update_weights()
        sc2 = self.eval()


def MLP_federated(clients_dataset):
    weights = None
    pre_scores = []
    post_scores = []
    MLP_classifiers = []
    for dataset in clients_dataset:
        X_train = dataset[0].to_numpy()
        y_train = dataset[2]
        X_test = dataset[1].to_numpy()
        y_test = dataset[3]
        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
        MLP_classifiers.append(clf)
        pre_scores.append(metrics(clf.predict(X_test),y_test))
        # Store the coefficientes of each client
        if weights is None:
            weights = clf.coefs_
        else:
            for i in range(len(weights)):
                weights[i] += clf.coefs_[i]

    # Compute the average of the stored coefficients
    for i in range(len(weights)):
        weights[i] = weights[i]/len(MLP_classifiers)

    for dataset, clf in zip(clients_dataset, MLP_classifiers):
        X_test = dataset[1].to_numpy()
        y_test = dataset[3]
        # Update the coefficientes of each client
        for i in range(len(weights)):
            clf.coefs_[i] = weights[i]
        post_scores.append(metrics(clf.predict(X_test),y_test))

    return pre_scores, post_scores
        