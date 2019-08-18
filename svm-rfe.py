# Generate data for SVM classifier with L1 regularization.
from __future__ import division

import numpy as np
import pandas as pd
import random
from cvxpy import *
from util import get_accuracy

np.random.seed(10)

nodes = 1
random.seed(10)

random_state = 23
num_features = 198


def solve_svm(X_train, y_train):
    C = 0.75
    num_feats = X_train.shape[1] - 1
    num_examples = len(y_train)
    beta = Variable(num_feats + 1, 1)  # x in our terminology
    epsil = Variable(num_examples, 1)
    g = C * norm(epsil, 1)
    for i in range(num_feats):
        g = g + 0.5 * square(beta[i])
    constraints = [epsil >= 0]
    for j in range(num_examples):
        constraints = constraints + [
            y_train[j] * (sum_entries(mul_elemwise(np.asmatrix(X_train[j]).T, beta))) >= 1 - epsil[j]]

    objective = Minimize(50 * g)
    prob = Problem(objective, constraints)
    try:
        result = prob.solve()
        if result is None:
            p = Problem(Minimize(52 * g), constraints)
            p.solve()
    except SolverError:
        try:
            print("Scaling Node: " + str(node_id))
            p = Problem(Minimize(52 * g), constraints)
            p.solve()
        except:
            pass

    return beta.value


def do_ref(num_features, X, y, X_test, y_test):
    s = [i for i in range(num_features)]
    r = []
    while len(s) is not 0:
        X_new = np.c_[X[:, s], X[:, -1]]
        w = solve_svm(X_new, y)
        w = np.asarray(w).reshape(-1)
        print(len(s))

        accuracy = get_accuracy(w, np.c_[X_test[:, s], X_test[:, -1]], y_test)
        print("Node Id: " + str(node_id) + ", Accuracy: " + str(accuracy))

        w = w[: -1]
        c = np.array([w_i**2 for w_i in w])
        c_list = list(c)
        idx = c.argmin()

        # Remove similar values across other places
        min_c = c[idx]
        min_real_indices = []
        min_indices = np.array(np.where(c == min_c)).reshape(-1)
        for ind in sorted(min_indices, reverse=True):
            s_removed = s.pop(ind)
            min_real_indices.append(s_removed)
            c_list.pop(ind)

        # s_f = s.pop(idx)
        # c_list.pop(idx)
        r = np.append(min_real_indices, r)
    return np.array(r, dtype=int)


from matplotlib import pyplot as plt
for node_id in range(nodes):
# node_id = 1
    dataset_dir = "/home/prabathabey/phd/tii-new/dataset/spoilt/" + str(random_state) + "/"
    df_training = pd.read_csv(dataset_dir + "train-" + str(node_id) + ".csv", index_col=False, dtype='float64')

    df_test = pd.read_csv(dataset_dir + "test-" + str(node_id) + ".csv", index_col=False, dtype='float64')

    X_train_int = df_training.drop(columns=[str(num_features)])
    y_train = np.array(df_training[str(num_features)])
    X_train = np.array(X_train_int)
    X_train = np.c_[X_train, np.ones(len(y_train))]

    X_test_int = df_test.drop(columns=[str(num_features)])
    y_test = np.array(df_test[str(num_features)])
    X_test = np.array(X_test_int)
    X_test = np.c_[X_test, np.ones(len(y_test))]

    rankings = do_ref(num_features, X_train, y_train, X_test, y_test)

    feature_counter = -1
    accuracies = []
    while feature_counter > (-1)*(len(rankings) + 1):
        sub_rankings = rankings[feature_counter:]
        w = solve_svm(np.c_[X_train[:, sub_rankings], X_train[:, -1]], y_train)
        acc = get_accuracy(w, np.c_[X_test[:, sub_rankings], X_test[:, -1]], y_test)
        accuracies.append(acc)
        feature_counter -= 1
        print(-1 * feature_counter)
    # w = solve_svm(X_train, y_train)
    x = [i + 1 for i in range(len(rankings))]
    y = accuracies
    print(len(x), len(y))
    plt.plot(x, y, c=np.random.rand(3,))
    plt.xlabel("Num of features")
    plt.ylabel("Accuracy - %")


plt.show()
