print("       (.)~(.)")
print("      (-------)")
print("-----ooO-----Ooo----")
print("    SELFMADE KNN")                                       
print("--------------------")
print("      ( )   ( )")
print("      /|\   /|\\")

import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import random
from matplotlib import style
from math import sqrt
from collections import Counter


def find_euclidean_dist(features, predictions):
    return np.linalg.norm(np.array(features) - np.array(predictions))


def find_k_nearest(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is less than total voting groups.")
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_dist = find_euclidean_dist(features, predict)
            distances.append([euclidean_dist, group])
    
    # Process of getting highest neighbor count
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]
    return vote_result, confidence


style.use("fivethirtyeight")

# DUMMY SET
# dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5], [7,7], [8,6]]}
# new_features = [5,7]
#
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.show()

accuracies = []
attempts = 50
for i in range(attempts):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df = df.replace('?', -99999).drop(['id'], 1)
    full_data = df.astype(float).values.tolist()
    # print(full_data[:5])
    # random.shuffle(full_data)
    # print(20*'#')
    # print(full_data[:5])

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    # Making predictions on the dataset
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = find_k_nearest(train_set, data, k=5)
            if group == vote:
                correct += 1
            # else:
            #     print(confidence)
            total += 1

    # print(f'Accuracy: {(correct/total) *100}%')
    accuracies.append(correct/total)

print(f"Average accuracy over {attempts} attempts: {(sum(accuracies)/len(accuracies))*100}%")