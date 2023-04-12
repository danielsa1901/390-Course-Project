# Needed libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from scipy.stats import skew

# add additional column in each csv to specify walking or jumping (for the training model), 0 or 1 has to be used to avoid issues with h5py
column_name = "WalkingJumping"
value_name = 0
value_name2 = 1

# Open the CSV file for reading and writing
with open('./Data/Daniel/Jumping Left pocket/Raw Data.csv', 'r') as infile:
    with open('./Data/Daniel/NewData/DJLP.csv', 'w') as outfile:
        # Create a CSV reader and writer objects
        reader = csv.reader(infile)
        writer = csv.writer(outfile, lineterminator='\n')
        # Read the header row from the input CSV file
        header_row = next(reader)
        # Add the new column title to the header row
        header_row.append(column_name)
        # Write the updated header row to the output CSV file
        writer.writerow(header_row)
        # Loop through each data row in the input CSV file
        for row in reader:
            #Add the new column value to the data row
            row.append(value_name2)
            #Write the updated data row to the output CSV file
            writer.writerow(row)

with open('./Data/Daniel/Jumping Right pocket/Raw Data.csv', 'r') as infile:
    with open('./Data/Daniel/NewData/DJRP.csv', 'w') as outfile:
        # Create a CSV reader and writer objects
        reader = csv.reader(infile)
        writer = csv.writer(outfile, lineterminator='\n')
        # Read the header row from the input CSV file
        header_row = next(reader)
        # Add the new column title to the header row
        header_row.append(column_name)
        # Write the updated header row to the output CSV file
        writer.writerow(header_row)
        # Loop through each data row in the input CSV file
        for row in reader:
            #Add the new column value to the data row
            row.append(value_name2)
            #Write the updated data row to the output CSV file
            writer.writerow(row)

with open('./Data/Daniel/Left pocket walking/Raw Data.csv', 'r') as infile:
    with open('./Data/Daniel/NewData/DWLP.csv', 'w') as outfile:
        # Create a CSV reader and writer objects
        reader = csv.reader(infile)
        writer = csv.writer(outfile, lineterminator='\n')
        # Read the header row from the input CSV file
        header_row = next(reader)
        # Add the new column title to the header row
        header_row.append(column_name)
        # Write the updated header row to the output CSV file
        writer.writerow(header_row)
        # Loop through each data row in the input CSV file
        for row in reader:
            #Add the new column value to the data row
            row.append(value_name)
            #Write the updated data row to the output CSV file
            writer.writerow(row)

with open('./Data/Daniel/Right pocket walking/Raw Data.csv', 'r') as infile:
    with open('./Data/Daniel/NewData/DWRP.csv', 'w') as outfile:
        # Create a CSV reader and writer objects
        reader = csv.reader(infile)
        writer = csv.writer(outfile, lineterminator='\n')
        # Read the header row from the input CSV file
        header_row = next(reader)
        # Add the new column title to the header row
        header_row.append(column_name)
        # Write the updated header row to the output CSV file
        writer.writerow(header_row)
        # Loop through each data row in the input CSV file
        for row in reader:
            #Add the new column value to the data row
            row.append(value_name)
            #Write the updated data row to the output CSV file
            writer.writerow(row)

# importing Data into python
G1Data = pd.read_csv(
    './Data/Daniel/NewData/DJLP.csv',sep=",")
G1Data2 = pd.read_csv(
    './Data/Daniel/NewData/DJRP.csv',sep=",")
G1Data3 = pd.read_csv(
    './Data/Daniel/NewData/DWLP.csv',sep=",")
G1Data4 = pd.read_csv(
    './Data/Daniel/NewData/DWRP.csv',sep=",")

#divided the data into 5 second segments, shuffling it, and splitting it into a ratio of 90:10 for training and testing
# divide each signal into 5 second windows
window_size = 5  # seconds
window_stride = 1  # second
G1Data_windows = []
G1Data2_windows = []
G1Data3_windows = []
G1Data4_windows = []

samples_per_window = int(window_size / G1Data['Time (s)'].diff().mean())

for i in range(0, len(G1Data), samples_per_window):
    # Get the current window of data
    window = G1Data.iloc[i:i+samples_per_window]
    # Check if the window contains enough samples
    if len(window) == samples_per_window:
        # Add the windowed data to the list
        G1Data_windows.append(window)

for i in range(0, len(G1Data2), samples_per_window):
    # Get the current window of data
    window = G1Data2.iloc[i:i+samples_per_window]
    # Check if the window contains enough samples
    if len(window) == samples_per_window:
        # Add the windowed data to the list
        G1Data2_windows.append(window)

for i in range(0, len(G1Data3), samples_per_window):
    # Get the current window of data
    window = G1Data3.iloc[i:i+samples_per_window]
    # Check if the window contains enough samples
    if len(window) == samples_per_window:
        # Add the windowed data to the list
        G1Data3_windows.append(window)

for i in range(0, len(G1Data4), samples_per_window):
    # Get the current window of data
    window = G1Data4.iloc[i:i+samples_per_window]
    # Check if the window contains enough samples
    if len(window) == samples_per_window:
        # Add the windowed data to the list
        G1Data4_windows.append(window)

#Combining the two Jumping window lists together, same applys for the walking sets
G1Data_windows.extend(G1Data2_windows)
G1Data3_windows.extend(G1Data4_windows)
#shuffling the data
random.shuffle(G1Data_windows)
random.shuffle(G1Data3_windows)
# Concatenate the windowed data into a new DataFrame
jumping_df = pd.concat(G1Data_windows)
walking_df = pd.concat(G1Data3_windows)
data = pd.concat([jumping_df, walking_df])
#splitting it 90:10
X = data.drop(columns=['WalkingJumping'])
y = data['WalkingJumping']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, stratify=None)

# putting data into the hdf5 file
with h5py.File('./hdf5_groups.h5', 'w') as hdf:
    G1 = hdf.create_group('Daniel')
    G1.create_dataset('Jumping Left', data=G1Data)
    G1.create_dataset('Jumping Right', data=G1Data2)
    G1.create_dataset('Walking Left', data=G1Data3)
    G1.create_dataset('Walking Right', data=G1Data4)
    G2 = hdf.create_group('/Josh')
    G3 = hdf.create_group('/Bradley')
    G4 = hdf.create_group('/Dataset')
    G5 = hdf.create_group('/Dataset/Testing')
    G5.create_dataset('training/data',data=X_train)
    G5.create_dataset('training/labels',data=y_train)
    G6 = hdf.create_group('/Dataset/Training')
    G6.create_dataset('testing/data',data=X_test)
    G6.create_dataset('testing/labels',data=y_test)

# Data Visualization
#skipped for now, more important stuff to work on first

# Pre-processing, applying an moving average filter to reduce noise
X_train_smoothed = X_train.rolling(window_size).mean()
X_test_smoothed = X_test.rolling(window_size).mean()

# Feature Extraction
def extract_features(window):
    features = []
    features.append(np.min(window))
    features.append(np.max(window))
    features.append(np.max(window) - np.min(window))
    features.append(np.mean(window))
    features.append(np.median(window))
    features.append(np.var(window))
    features.append(skew(window))
    return features

X_train_features = [extract_features(window) for window in X_train_smoothed.values]
X_test_features = [extract_features(window) for window in X_test_smoothed.values]

train_df = pd.DataFrame(X_train_features)
train_df = train_df.dropna()
X_train_features = train_df.to_numpy()

test_df = pd.DataFrame(X_test_features)
test_df = test_df.dropna()
X_test_features = test_df.to_numpy()

y_train = y_train[:-4]
y_test = y_test[:-4]

#Classifier
# Train the logistic regression model
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train_features, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_features)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

