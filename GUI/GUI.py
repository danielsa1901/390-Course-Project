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
    img = tk.PhotoImage(file="bg-dark-grey.png")

    style = darkstyle(window)


    lab = ttk.Label(
        window,
        text="Hello World",
        compound="center",
        font="arial 50",
        image=img)
    lab.pack(fill="both", expand=1)


    button = ttk.Button(
        window,
        text="plot",
        command = plot,
        style="Accentbutton"
        )

    button.place(relx=0.43, rely=0.7, width=100, height=30)


    window.mainloop()

main_window()