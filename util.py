import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def get_accuracy(w, X_test, y_test):
    y_pred = []
    for i in range(0, len(X_test)):
        a = np.dot(X_test[i], w)
        if a < 0:
            y_pred.append(-1)
        else:
            y_pred.append(1)

    c = 0
    for i in range(0, len(y_test)):
        if y_test[i] == y_pred[i]:
            c += 1

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    accuracy = c / float(len(y_test))
    # print("Node Id: " + str(node_id) + ", Accuracy: " + str(accuracy))
    return accuracy