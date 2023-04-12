# Needed libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

# add additional column in each csv to specify walking or jumping (for the training model)
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
    G6 = hdf.create_group('/Dataset/Training')


# Data Visualization


# Pre-processing
