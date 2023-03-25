# Needed libraries
import numpy as py
import h5py
import matplotlib.pyplot as plt
import pandas as pd

# importing Data into python
G1Data = pd.read_csv(
    './Data/Daniel/Jumping Left pocket/Raw Data.csv').iloc[:, :]
G1Data2 = pd.read_csv(
    './Data/Daniel/Jumping Right pocket/Raw Data.csv').iloc[:, :]
G1Data3 = pd.read_csv(
    './Data/Daniel/Left pocket walking/Raw Data.csv').iloc[:, :]
G1Data4 = pd.read_csv(
    './Data/Daniel/Right pocket walking/Raw Data.csv').iloc[:, :]

# putting data into the hdf5 file
with h5py.File('./hdf5_groups.h5', 'w') as hdf:
    G1 = hdf.create_group('/Daniel')
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
