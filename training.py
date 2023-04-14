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
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve
from scipy.stats import kurtosis

# for output visualization
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)

window_size = 5  # seconds

# add additional column in each csv to specify walking or jumping (for the training model), 0 or 1 has to be used to avoid issues with h5py
column_name = "WalkingJumping"
value_name = 0
value_name2 = 1

# Open the CSV file for reading and writing
def extract_data(in_file_path, out_file_path, column_name, dummy_value):
    with open(in_file_path, 'r') as infile, \
        open(out_file_path, 'w') as outfile:
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
            row.append(dummy_value)
            #Write the updated data row to the output CSV file
            writer.writerow(row)

# Segments data into 5 second windows        
def split_data_into_windows(group_data, group_windows, samples_per_window):
    for i in range(0, len(group_data), samples_per_window):
        # Get the current window of data
        window = group_data.iloc[i:i+samples_per_window]
        # Check if the window contains enough samples
        if len(window) == samples_per_window:
            # Add the windowed data to the list
            group_windows.append(window)

#test Data
extract_data('./Data/TestData1/Jumping1.csv', './Data/TestData1/NewJumping1.csv', column_name, value_name2)
extract_data('./Data/TestData2/Jumping2.csv', './Data/TestData2/NewJumping2.csv', column_name, value_name2)
extract_data('./Data/TestData3/Jumping3.csv', './Data/TestData3/NewJumping3.csv', column_name, value_name2)
extract_data('./Data/TestData4/Jumping4.csv', './Data/TestData4/NewJumping4.csv', column_name, value_name2)
extract_data('./Data/TestData5/Walking1.csv', './Data/TestData5/NewWalking1.csv', column_name, value_name)
extract_data('./Data/TestData6/Walking2.csv', './Data/TestData6/NewWalking2.csv', column_name, value_name)
extract_data('./Data/TestData7/Walking3.csv', './Data/TestData7/NewWalking3.csv', column_name, value_name)
extract_data('./Data/TestData8/Walking4.csv', './Data/TestData8/NewWalking4.csv', column_name, value_name)

# importing Data into python
# Jumping
G1Data1 = pd.read_csv(
    './Data/TestData1/NewJumping1.csv',sep=",")
G1Data2 = pd.read_csv(
    './Data/TestData2/NewJumping2.csv',sep=",")
G1Data3 = pd.read_csv(
    './Data/TestData3/NewJumping3.csv',sep=",")
G1Data4 = pd.read_csv(
    './Data/TestData4/NewJumping4.csv',sep=",")
# Walking
G2Data1 = pd.read_csv(
    './Data/TestData5/NewWalking1.csv',sep=",")
G2Data2 = pd.read_csv(
    './Data/TestData6/NewWalking2.csv',sep=",")
G2Data3 = pd.read_csv(
    './Data/TestData7/NewWalking3.csv',sep=",")
G2Data4 = pd.read_csv(
    './Data/TestData8/NewWalking4.csv',sep=",")

#divided the data into 5 second segments, shuffling it, and splitting it into a ratio of 90:10 for training and testing
#smooth and normalize data first, then split it up
scaler = StandardScaler()

def SmoothNormalize(dataframe):
    otherColumns = dataframe[['Time (s)', 'WalkingJumping']]
    data = dataframe[['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']]
    data_smoothed = data.rolling(window=5).mean().dropna() #moving average filter
    data_normalized = scaler.fit_transform(data_smoothed) #normalize
    df_smoothed_normalized = pd.DataFrame(data=data_normalized, columns=data.columns) #add time and WalkingJumping column back in
    df_smoothed_normalized[['Time (s)', 'WalkingJumping']] = otherColumns[len(data) - len(data_smoothed):] #add time and WalkingJumping column back in
    return df_smoothed_normalized

jump1Smooth = SmoothNormalize(G1Data1).dropna()
jump2Smooth = SmoothNormalize(G1Data2).dropna()
jump3Smooth = SmoothNormalize(G1Data3).dropna()
jump4Smooth = SmoothNormalize(G1Data4).dropna()
walk1Smooth = SmoothNormalize(G2Data1).dropna()
walk2Smooth = SmoothNormalize(G2Data2).dropna()
walk3Smooth = SmoothNormalize(G2Data3).dropna()
walk4Smooth = SmoothNormalize(G2Data4).dropna()

# divide each signal into 5 second windows
window_stride = 1  # second
#jumping
G1Data1_windows = []
G1Data2_windows = []
G1Data3_windows = []
G1Data4_windows = []
#walking
G2Data1_windows = []
G2Data2_windows = []
G2Data3_windows = []
G2Data4_windows = []

#Jumping
samples_per_window = int(window_size / G1Data1['Time (s)'].diff().mean())
split_data_into_windows(jump1Smooth, G1Data1_windows, samples_per_window)
split_data_into_windows(jump2Smooth, G1Data2_windows, samples_per_window)
split_data_into_windows(jump3Smooth, G1Data3_windows, samples_per_window)
split_data_into_windows(jump4Smooth, G1Data4_windows, samples_per_window)
#Walking
samples_per_window = int(window_size / G2Data1['Time (s)'].diff().mean())
split_data_into_windows(walk1Smooth, G2Data1_windows, samples_per_window)
split_data_into_windows(walk2Smooth, G2Data2_windows, samples_per_window)
split_data_into_windows(walk3Smooth, G2Data3_windows, samples_per_window)
split_data_into_windows(walk4Smooth, G2Data4_windows, samples_per_window)

#Combining the two Jumping window lists together, same applys for the walking sets

jumping_list = []
walking_list = []

for lst in [G1Data1_windows, G1Data2_windows, G1Data3_windows, G1Data4_windows]:
    jumping_list.extend(lst)
for lst in [G2Data1_windows, G2Data2_windows, G2Data3_windows, G2Data4_windows]:
    walking_list.extend(lst)

#shuffling the data
full_list = []

for lst in [jumping_list, walking_list]:
    full_list.extend(lst)
random.shuffle(full_list)
# Concatenate the windowed data into a new DataFrame
data = pd.concat(full_list)

#splitting it 90:10
X = data.drop(columns=['WalkingJumping'])
y = data['WalkingJumping']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False, stratify=None)

# # putting data into the hdf5 file
# with h5py.File('./hdf5_groups.h5', 'w') as hdf:
#     G1 = hdf.create_group('Daniel')
#     G1.create_dataset('Jumping Left', data=G1Data1)
#     G1.create_dataset('Jumping Right', data=G1Data2)
#     G1.create_dataset('Walking Left', data=G1Data3)
#     G1.create_dataset('Walking Right', data=G1Data4)
#     G2 = hdf.create_group('/Josh')
#     G2.create_dataset('Jumping Left', data=G2Data1)
#     G2.create_dataset('Jumping Right', data=G2Data2)
#     G2.create_dataset('Walking Left', data=G2Data3)
#     G2.create_dataset('Walking Right', data=G2Data4)
#     G3 = hdf.create_group('/Bradley')
#     G4 = hdf.create_group('/Dataset')
#     G5 = hdf.create_group('/Dataset/Testing')
#     G5.create_dataset('training/data',data=X_train)
#     G5.create_dataset('training/labels',data=y_train)
#     G6 = hdf.create_group('/Dataset/Training')
#     G6.create_dataset('testing/data',data=X_test)
#     G6.create_dataset('testing/labels',data=y_test)

# Data Visualization
#skipped for now, more important stuff to work on first

scaler = StandardScaler()
l_reg = LogisticRegression(max_iter=10000)

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
    features.append(np.std(window))
    features.append(np.sqrt(np.mean(np.square(window))))
    features.append(kurtosis(window))
    return features

X_train_features = [extract_features(window) for window in X_train.values]
X_test_features = [extract_features(window) for window in X_test.values]

#Classifier
# Train the logistic regression model
clf = make_pipeline(scaler, l_reg)
clf.fit(X_train_features, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_features)
y_clf_prob = clf.predict_proba(X_test_features)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
#print('y_pred: ', y_pred)
#print('y_clf_prob: ', y_clf_prob)
print("Accuracy: {:.2f}%".format(accuracy * 100))
# recall = recall_score(y_test, y_pred) 
# print('Recall: ', recall)

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

# auc = roc_auc_score(y_test, y_clf_prob[:, 1])
# print('AUC: ', auc)

# f1score = f1_score(y_test, y_pred)
# print('F1 Score: ', f1score)