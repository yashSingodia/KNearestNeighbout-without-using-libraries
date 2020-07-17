#!/usr/bin/env python
# coding: utf-8


# K Nearest neighbour without using scikit-learn 

import pandas as pd
import numpy as np
from math import sqrt



# Calculate Eucledian distance between two vectors row1 and row2
def eucledian(row1, row2):
    dist = 0.0
    for i in range(len(row1) - 1):
        dist += (row1[i] - row2[i])**2
    return sqrt(dist)




# Find k nearest neighbour to each test_row in test dataset
def find_nearest_neighbours(train, test_row, k):
    distances = []
    
    for train_row in train:
        dist = eucledian(train_row, test_row)
        distances.append((train_row, dist))
    distances.sort(key = lambda x : x[1])  # Sorting by dist using lambda function
    
    neighbours = []
    for i in range (k):
        neighbours.append(distances[i][0])
    return neighbours
        
    



# Predict class based on each test_row by finding k nearest neighbours to each test_row
def predict_class(train, test_row, k):
    neighbours = find_nearest_neighbours(train, test_row, k)
    output_values = [row[-1] for row in neighbours]
    
    predict_class = max(set(output_values), key = output_values.count )
    return predict_class



# KNN algorithm
def k_nearest_neighbour(train, test, k):
    predictions = []
    
    for row in test:
        out = predict_class(train, row, k)
        predictions.append(out)
    
    return (predictions)






