import tkinter as tk
import tkinter.ttk as ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

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
        filename = tk.filedialog.askopenfilename(initialdir = "/",
                                            title = "Select a File",
                                            filetypes = (("CSV files",
                                                            "*.csv*"),
                                                        ("all files",
                                                            "*.*")))
      
        # Change label contents
        label_file_explorer.configure(text="File Opened: "+filename)

        with open(filename, 'r') as csv_file:
            stuff = csv_file.read()
            print(stuff)

    # plot function is created for plotting the graph in tkinter window
    def plot():
        # the figure that will contain the plot
        fig = Figure(figsize = (5, 5),
                    dpi = 100)
    
        # list of squares
        y = [i**2 for i in range(101)]
    
        # adding the subplot
        plot1 = fig.add_subplot(111)
    
        # plotting the graph
        plot1.plot(y)
    
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
    window.resizable(False, False)
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
        image=img
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
    label_plot.pack(fill="both", expand=1)
    button_explore.place(relx=0.43, rely=0.4, width=100, height=30)
    button_plot.place(relx=0.43, rely=0.9, width=100, height=30)


    window.mainloop()

main_window()