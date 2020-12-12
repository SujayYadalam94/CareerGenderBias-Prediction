from __future__ import division
import sys
import os
import numpy as np
import random
import csv
import time
from math import floor,sqrt,exp,pi
from collections import Counter

# Global Variables
features = []
freq_table = {}
dist_X0 = {}
dist_X1 = {}
p0 = 1
p1 = 1

def create_input(input_csv):
    global features

    out_file = open("pre-processed.csv", "w")
    reader = csv.reader(open(input_csv,"rb"), delimiter=",")
    writer = csv.writer(out_file, delimiter=',')

    for sample in reader:
        remove = False
        for feat in sample:
            if feat == '':
                remove = True
        if not remove:
            writer.writerow(sample)

    out_file.close() 
    reader = csv.reader(open("pre-processed.csv", "rb"), delimiter=",")
    os.remove("pre-processed.csv")
    features = next(reader)
    X = list(reader)
    X = np.array(X).astype("float")

    # Swap dscore to the first column
    idx = features.index("dscore")
    features[0], features[idx] = features[idx], features[0]
    X[:,0], X[:,idx] = X[:,idx], X[:,0].copy()

    # Convert dscore to classes
    col = X[:,0]
    i = 0
    while i < len(col):
        if col[i] >= 0.65:
            col[i] = 1
        else:
            col[i] = 0
        i += 1

    # Convert birthyear, countrycit_num and countryres_num columns
    c = features.index("birthyear")
    col = X[:,c]
    curr_year = np.full((col.shape), 2020)
    age = curr_year - col
    i = 0
    while i < len(age):
        if age[i] < 25:
            X[i,c] = 25
        elif age[i] < 40:
            X[i,c] = 40
        else:
            X[i,c] = 50
        i += 1

    c = features.index("countrycit_num")
    col = X[:,c]
    i = 0
    while i < len(col):
        if col[i] == 11723:
            col[i] = 1
        else:
            col[i] = 0
        i += 1

    c = features.index("countryres_num")
    col = X[:,c]
    i = 0
    while i < len(col):
        if col[i] == 12310:
            col[i] = 1
        else:
            col[i] = 0
        i += 1

    return X

def prepare_distributions(X):
    i = 0
    p = {}
    while i < len(X[0]):
        feat = features[i+1]
        col = X[:,i]
        # Test metrics modelled as Gaussian
        if feat.find("Met:") != -1:
            # For Gaussian, we need avg and var
            l = ["G"]
            l.append(np.average(col))
            l.append(np.var(col, ddof=len(X)-1))
        # Other features modelled as multinomial
        else:
            # For Multinomial, we need counts and total
            l = ["M"]
            counter = Counter(col)
            l.append(counter)
            l.append(len(X))

        p[feat] = l
        i += 1

    return p

def calc_feat_prob(feat, x, y):
    global dist_X0
    global dist_X1

    l = []
    if y == 0:
        l = dist_X0[feat]
    else:
        l = dist_X1[feat]

    # Gaussian distribution
    if l[0] == "G":
        mean = l[1]
        var = l[2]
        prob = (1.0 / sqrt(2*pi*var)) * exp(-((x-mean)**2/(2*var)))
        return prob
    # Multinomial distribution
    elif l[0] == "M":
        count = l[1][x]
        total = l[2]
        return ((count+1)/(total+len(l[1])))

def calc_cond_prob(x_new, y):
    p = 1.0
    i = 0
    while i < len(x_new):
        feat = features[i+1] # dscore is part of features
        x = x_new[i]
        # Convert these 3 features into multinomial
        if feat == "birthyear":
            if x < 25:
                x = 25
            elif x < 40:
                x = 40
            else:
                x = 50
        elif feat == "countrycit_num":
            if x == 11723:
                x = 1
            else:
                x = 0
        elif feat == "countryres_num":
            if x == 12310:
                x = 1
            else:
                x = 0
        temp = calc_feat_prob(feat, x, y)
        p *= temp
        i += 1

    return p
    
def naive_bayes_prepare(X):
    global dist_X0
    global dist_X1
    global p0
    global p1

    X_sorted = X[np.argsort(X[:, 0])]
    split = np.searchsorted(X_sorted[:,0], 1)

    X_0 = X_sorted[:split,:]
    X_1 = X_sorted[split:,:]

    dist_X0 = prepare_distributions(X_0[:,1:])
    dist_X1 = prepare_distributions(X_1[:,1:])

    p0 = len(X_0)/len(X_sorted)
    p1 = len(X_1)/len(X_sorted)

def naive_bayes_output(x_new):
    global p0
    global p1

    p_x_y0 = calc_cond_prob(x_new, 0)
    p_x_y1 = calc_cond_prob(x_new, 1)

    if p0 * p_x_y0 >= p1 * p_x_y1:
        return 0
    else:
        return 1

def create_folds(X, num_folds):
    Xsort = X[np.argsort(X[:,0])]
    split = np.searchsorted(Xsort[:,0], 1)
    X0 = Xsort[0:split]
    X1 = Xsort[split:]
    len_X0 = int(len(X0) / num_folds)
    len_X1 = int(len(X1) / num_folds)
    
    folds = []
    for i in range(num_folds):
        fold = np.vstack((X0[i*len_X0:(i+1)*len_X0], X1[i*len_X1:(i+1)*len_X1]))
        folds.append(fold)
    fold = np.vstack((X0[i*len_X0:],X1[i*len_X1:]))
    folds.append(fold)
    return folds

if __name__ == "__main__": 

    if len(sys.argv) < 2:
        print("Usage: python naive_bayes.py input_file")
        exit(1)

    start = time.time()

    # Read the input from csv and pre-process it
    X = create_input(sys.argv[1])

    # Cross Validation
    num_folds = 10
    folds = create_folds(X, num_folds)
    j = 0
    total_correct = 0
    total_input_size = 0

    for j in range(num_folds):
        test_set = folds[j]
        train_set = folds[num_folds]
        correct = 0
        i = 0
        while i < len(folds):
            if i != j:
                train_set = np.vstack((train_set, folds[i]))
            i += 1

        # Pre-calculate the probability distributions
        naive_bayes_prepare(train_set)
        for row in test_set:
            # Predict using naive bayes
            y = naive_bayes_output(row[1:])
            if y == row[0]:
                correct += 1

        print ("Accuracy = %f" % (correct/len(folds[j])))
        total_correct += correct
        total_input_size += len(folds[j])

    print ("TOTAL ACCURACY = %f" % (total_correct/total_input_size))
    end = time.time()
    total_time = end - start
    print ("TOTAL TIME ELAPSED: %f" % total_time)
