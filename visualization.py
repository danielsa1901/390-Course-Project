import matplotlib.pyplot as plt
import pandas as pd

def  continuous_density_plot(file_path):
    dataset = pd.read_csv(file_path)

    data = dataset.iloc[:, 0:5]

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))

    data.plot(ax=ax.flatten()[0:5], kind='density', subplots=True, sharex=False)
    fig.tight_layout()
    plt.show()

def  scatter_matrix_plot(file_path):
    dataset = pd.read_csv(file_path)

    data = dataset.iloc[:, 0:5]

    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(30, 30))

    pd.plotting.scatter_matrix(data, ax=ax, s=5)
    fig.tight_layout()
    plt.show()

def scatter_plot(x_data, y_data, x_title="x_axis", y_title="y_axis"):
    plt.plot(x_data, y_data)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

def get_two_columns(pd_dataframe, col1_index, col2_index):
    col1_values = pd_dataframe.iloc[:, col1_index].values
    col2_values = pd_dataframe.iloc[:, col2_index].values
    return col1_values, col2_values

# continuous_density_plot("Data\Josh\jump LP\Raw Data.csv")
# scatter_matrix_plot("Data\Josh\walk LP\Raw Data.csv")

# x_values = [1, 2, 3, 4, 5]
# y_values = [10, 8, 6, 4, 2]
# x_title = "X Values"
# y_title = "Y Values"
# scatter_plot(x_values, y_values, "Time", "Acceleration")

# df = pd.DataFrame({
#     'column1': [1, 2, 3, 4, 5],
#     'column2': [9, 8, 6, 4, 2],
#     'column3': ['a', 'b', 'c', 'd', 'e']
# })

# col1_idx = 0
# col2_idx = 1

# col1_values, col2_values = get_two_columns(df, col1_idx, col2_idx)

# print(col1_values)  # prints: [1 2 3 4 5]
# print(col2_values)  # prints: [10 8 6 4 2]