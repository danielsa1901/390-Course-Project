import tkinter as tk
import tkinter.ttk as ttk
import joblib
import csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler

def get_two_columns(pd_dataframe, col1_index, col2_index):
    col1_values = pd_dataframe.iloc[:, col1_index].values
    col2_values = pd_dataframe.iloc[:, col2_index].values
    return col1_values, col2_values

def SmoothNormalize(dataframe):
    scaler = StandardScaler()
    otherColumns = dataframe[['Time (s)']]
    data = dataframe[['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']]
    data_smoothed = data.rolling(window=50).mean().dropna() #moving average filter
    data_normalized = scaler.fit_transform(data_smoothed) #normalize
    df_smoothed_normalized = pd.DataFrame(data=data_normalized, columns=data.columns) #add time and WalkingJumping column back in
    df_smoothed_normalized[['Time (s)']] = otherColumns[len(data) - len(data_smoothed):] #add time and WalkingJumping column back in
    return df_smoothed_normalized
# Feature Extraction
def extract_features(window):
    features = []
    x_acc = window["Linear Acceleration x (m/s^2)"]
    y_acc = window["Linear Acceleration y (m/s^2)"]
    z_acc = window["Linear Acceleration z (m/s^2)"]
    abs_acc = window["Absolute acceleration (m/s^2)"]
    features.append(np.min(x_acc))
    features.append(np.min(y_acc))
    features.append(np.min(z_acc))
    features.append(np.max(x_acc))
    features.append(np.max(y_acc))
    features.append(np.max(z_acc))
    features.append(np.max(x_acc)-np.min(x_acc))
    features.append(np.max(y_acc)-np.min(y_acc))
    features.append(np.max(z_acc)-np.min(z_acc))
    features.append(np.mean(x_acc))
    features.append(np.mean(y_acc))
    features.append(np.mean(z_acc))
    features.append(np.median(x_acc))
    features.append(np.median(y_acc))
    features.append(np.median(z_acc))
    features.append(np.var(x_acc))
    features.append(np.var(y_acc))
    features.append(np.var(z_acc))
    features.append(skew(x_acc))
    features.append(skew(y_acc))
    features.append(skew(z_acc))
    features.append(np.std(x_acc))
    features.append(np.std(y_acc))
    features.append(np.std(z_acc))
    features.append(np.sqrt(np.mean(np.square(x_acc))))
    features.append(np.sqrt(np.mean(np.square(y_acc))))
    features.append(np.sqrt(np.mean(np.square(z_acc))))
    features.append(kurtosis(x_acc))
    features.append(kurtosis(y_acc))
    features.append(kurtosis(z_acc))
    #test
    features.append(np.min(abs_acc))
    features.append(np.max(abs_acc))
    features.append(np.max(abs_acc)-np.min(abs_acc))
    features.append(np.mean(abs_acc))
    features.append(np.median(abs_acc))
    features.append(np.var(abs_acc))
    features.append(skew(abs_acc))
    features.append(np.std(abs_acc))
    features.append(np.sqrt(np.mean(np.square(abs_acc))))
    features.append(kurtosis(abs_acc))
    return features

def preprocess_data(data):
    scaler = StandardScaler()
    clean_data = SmoothNormalize(data).dropna()
    return clean_data

def split_into_windows(data, samples_per_window):

    i=0
    data_windows=[]
    while i < len(data):
        window = data.iloc[i:i+samples_per_window]
        data_windows.append(window)
        i=i+samples_per_window
    return data_windows

def get_features(data_windows):
    features = [extract_features(window) for window in data_windows]
    return features

def run_model(data):
    model = joblib.load(r'C:\Users\Josh\Brogramming\ELEC390\project\390-Course-Project\test2.joblib')
    predictions = model.predict(data)
    return predictions

def create_output_csv(in_file_path, out_file_path, predictions, samples_per_window):
    with open(in_file_path, 'r') as infile, \
    open(out_file_path, 'w') as outfile:
        # Create a CSV reader and writer objects
        reader = csv.reader(infile)
        writer = csv.writer(outfile, lineterminator='\n')
        # Read the header row from the input CSV file
        header_row = next(reader)
        # Add the new column title to the header row
        header_row.append("walk-jump")
        # Write the updated header row to the output CSV file
        writer.writerow(header_row)

        # Loop through each data row in the input CSV file
        for index, row in enumerate(reader):
            if (index == 0):
                print(row)
            #calculate what prediction to append
            prediction_index = index // samples_per_window
            #Add the new column value to the data row
            row.append(int(predictions[prediction_index]))
            #Write the updated data row to the output CSV file
            writer.writerow(row)

def darkstyle(window):
    ''' Return a dark style to the window'''
    
    style = ttk.Style(window)
    window.tk.call('source', 'azure dark/azure dark.tcl')
    style.theme_use('azure')
    style.configure("Accentbutton", foreground='white')
    style.configure("Togglebutton", foreground='white')
    return style

def main_window():
    # Function for opening the
    # file explorer window
    def browseFiles():
        filename = tk.filedialog.askopenfilename(initialdir = r"C:\Users\Josh\Brogramming\ELEC390\project\390-Course-Project",
                                            title = "Select a File",
                                            filetypes = (("CSV files",
                                                            "*.csv*"),
                                                        ("all files",
                                                            "*.*")))
      
        # Change label contents
        label_file_explorer.configure(text="File Opened: "+ filename +"\n\nModel Classified the CSV File\n\nsaved output to output123.csv")

        #get csv file and read into pd dataframe
        data = pd.read_csv(filename, sep=",")

        #preprocess data using same technique as training
        clean_data = preprocess_data(data)
        window_size = 5  # seconds
        samples_per_window = int(window_size / clean_data['Time (s)'].diff().mean())
        windows = split_into_windows(clean_data, samples_per_window)
        features = get_features(windows)
        #run model
        print("Running Model:")
        predictions = run_model(features)
        print("Model finished")
        create_output_csv(filename, "output123.csv", predictions, samples_per_window)
        # with open(filename, 'r') as csv_file:
        #     data = csv_file.read()

    # plot function is created for plotting the graph in tkinter window
    def plot():
        # the figure that will contain the plot
        fig = Figure(figsize = (5, 5),
                    dpi = 100)
    
        # adding the subplot
        plot1 = fig.add_subplot(111)

        dataset = pd.read_csv("output123.csv")
        col1, col2 = get_two_columns(dataset, 0, 5)
        plot1.set_xlabel("Time")
        plot1.set_ylabel("Walk(0) / Jump(1)")
        plot1.set_title("Walk/Jump")
        # plotting the graph
        plot1.plot(col1, col2)
    
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig,
                                master = window)  
        canvas.draw()
    
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()
    
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas,
                                    window)
        toolbar.update()
    
        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()

    """ The window with the darkstyle """
    window = tk.Tk()
    window.title("My App")
    window.geometry("1280x720")
    window.resizable(True, True)

    #Set window background color
    window.config(background = "white")
    img = tk.PhotoImage(file="bg-dark-grey.png")

    style = darkstyle(window)

    # Create a File Explorer label
    label_file_explorer = tk.Label(
        window,
        text="File Explorer",
        compound="center",
        font="arial 12",
        )

    label_plot = ttk.Label(
        window,
        text="Plot",
        compound="center",
        font="arial 12",
        image=img
        )

    button_explore = ttk.Button(
        window,
        text = "Browse Files",
        command = browseFiles,
        style="Accentbutton"
        )

    button_plot = ttk.Button(
        window,
        text="plot",
        command = plot,
        style="Accentbutton"
        )

    label_file_explorer.pack(fill="both", expand=1)
    button_explore.place(relx=0.45, rely=0.7, width=128, height=36)
    button_plot.place(relx=0.45, rely=0.8, width=128, height=36)

    window.mainloop()

main_window()