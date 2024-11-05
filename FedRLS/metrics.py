import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score, jaccard_score, accuracy_score

def metrics(y_predicted, y_test, average="micro"):
    """
    Computes performance, f1 score, precision, recall, jaccard and accuracy given y_p and y_t
    input:
      y_predicted: list
      y_test: list
    return:
      dict -> performance, f1 score, precision, recall, jaccard and accuracy
    """
    performance = np.mean(np.equal(y_predicted, y_test))
    f1 = f1_score(y_predicted, y_test, average=average) # binary
    precision = precision_score(y_predicted, y_test, average=average)
    recall = recall_score(y_predicted, y_test, average=average)
    jaccard = jaccard_score(y_predicted, y_test, average=average)
    accuracy = accuracy_score(y_predicted, y_test)
    
    train_performance = {"f1": f1, "precision": precision, "recall":recall, "jaccard":jaccard, "accuracy":accuracy, "performance":performance}

    return train_performance