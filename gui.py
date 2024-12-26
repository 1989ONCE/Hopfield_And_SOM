import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageGrab
import pygetwindow
import pyautogui
import platform
import time
from SOM import SOM
from Hopfield import Hopfield

def str2float(strlist):
    return [round(float(i)) if float(i).is_integer() else float(i) for i in strlist]

class gui():
    def __init__(self, app_name, app_width, app_height):
        self.file_name = ''
        self.data = None
        self.epoch = 0
        self.lr = 0.1
        

        # container initialization
        self.container = tk.Tk()
        self.container.config(bg='white', padx=10, pady=10)
        self.container.maxsize(app_width, app_height)
        self.container.title(app_name)
        self.container.geometry(str(app_width) + 'x' + str(app_height))


        # components initialization
        self.graph_frame = tk.Frame(self.container, width=830, height=740, bg='red')
        self.setting_frame = tk.Frame(self.container, width=500, height=740, bg='black')

        self.canvas = FigureCanvasTkAgg(master = self.graph_frame)
        self.canvas.get_tk_widget().config(width=800, height=700)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Ensure the canvas is packed correctly

        self.canvas_2 = FigureCanvasTkAgg(master = self.setting_frame)
        self.canvas_2.get_tk_widget().config(width=500, height=740)

        self.is_hopfield = True
        # Define Our Images
        self.imgHopfield = tk.PhotoImage(file = self.get_path("data/img"+"/hopfield.png"))
        self.imgSOM = tk.PhotoImage(file = self.get_path("data/img"+"/som.png"))
        
        # Datasets Option
        self.hopfieldOption = self.get_all_files_name('data/hopfield')
        self.hopfieldOption.remove('Bonus_Testing.txt')
        self.hopfieldOption.remove('Basic_Testing.txt')
        self.somOption = self.get_all_files_name('data/som')

        # Create A Button
        self.toggle_canvas = tk.Canvas(
            self.setting_frame,
            width=self.imgHopfield.width(),  # Set canvas size to match the image
            height=self.imgHopfield.height(),
            highlightthickness=0,  # Remove the border of the canvas
            bg='white'
        )
        self.toggle_canvas.create_image(0, 0, anchor='nw', image=self.imgHopfield)
        self.toggle_canvas.bind("<Button-1>", lambda _: self.switch())

        
        self.dataDropDown = ttk.Combobox(master = self.setting_frame,
                        values=self.hopfieldOption)
        
        self.dim_label = tk.Label(self.setting_frame, text='Pattern Size', bg='white')
        self.dim_text = tk.Label(self.setting_frame, text='-', bg='white')
        self.sample_num_label = tk.Label(self.setting_frame, text='Numbers of Samples: ', bg='white')
        self.sample_num = tk.Label(self.setting_frame, text='-', bg='white')
        self.epoch_label = tk.Label(self.setting_frame, text='Epoch:', bg='white')
        self.epoch_box = tk.Spinbox(self.setting_frame, increment=1, from_=0, width=5, bg='white', textvariable=tk.StringVar(value='100'))
        self.lrn_rate_label = tk.Label(self.setting_frame, text='Learning Rate:', bg='white')
        self.lrn_rate_box = tk.Spinbox(self.setting_frame,  format="%.2f", increment=0.01, from_=0.0,to=1, width=5, bg='white', textvariable=tk.StringVar(value='0.01'))
        self.map_size_label = tk.Label(self.setting_frame, text='Map Size:', bg='white')
        self.map_size_text = tk.Label(self.setting_frame, text='-', bg='white')
        self.sigma_label = tk.Label(self.setting_frame, text='Sigma:', bg='white')
        self.sigma_box = tk.Spinbox(self.setting_frame,  format="%.2f", increment=0.1, from_=0.0, width=5, bg='white', textvariable=tk.StringVar(value='3.0'))
        self.train_btn = tk.Button(master = self.setting_frame,  
                     command = self.train_model, 
                     height = 2,  
                     width = 10, 
                     text = "Train Hopfield",
                     highlightbackground='white') 
        self.save_graph_frame_btn = tk.Button(master = self.setting_frame,  
                     command = self.save_graph_frame, 
                     height = 2,  
                     width = 16, 
                     text = "Take screenshot",
                     highlightbackground='white')

        # components placing
        self.setting_frame.place(x=5, y=20)
        self.graph_frame.place(x=515, y=5)
        self.canvas.get_tk_widget().place(x=10, y=30)


        self.figure = None
        self.figure_2 = None
        # toggle and data dropdown
        self.toggle_canvas.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.dataDropDown.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.dataDropDown.set('Select a Dataset')
        self.dataDropDown.bind("<<ComboboxSelected>>",  lambda e: self.load(self.dataDropDown.get()))

        self.dim_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.dim_text.grid(row=2, column=1, padx=5, pady=5, sticky='w')   
        self.sample_num_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.sample_num.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.epoch_label.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.epoch_box.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        self.lrn_rate_label.grid(row=6, column=0, padx=5, pady=5, sticky='w')
        self.lrn_rate_box.grid(row=6, column=1, padx=5, pady=5, sticky='w')
        self.train_btn.grid(row=8, column=0, padx=5, pady=5, sticky='w')
        self.save_graph_frame_btn.grid(row=8, column=1, padx=5, pady=5, sticky='w')
        self.canvas_2.get_tk_widget().grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky='w')

    
    # Determine is hopfield or SOM
    def switch(self):
        self.data = None
        # clear canvas 1 and 2
        if self.figure:
            self.figure.clf()
            self.canvas.draw()
        if self.figure_2:
            self.figure_2.clf()
            self.canvas_2.draw()

        if self.is_hopfield:
            self.toggle_canvas.create_image(0, 0, anchor='nw', image=self.imgSOM)
            self.is_hopfield = False
            self.dataDropDown.config(values=self.somOption)
            self.sample_num.config(text='-')
            self.dim_label.config(text='Dim: ')
            self.dim_text.config(text="-")
            self.train_btn.config(text='Train SOM')
            self.map_size_label.grid(row=4, column=0, padx=5, pady=5, sticky='w')
            self.map_size_text.grid(row=4, column=1, padx=5, pady=5, sticky='w')
            self.map_size_label.config(text='Map Size: ')
            self.map_size_text.config(text='-')
            self.sigma_label.grid(row=7, column=0, padx=5, pady=5, sticky='w')
            self.sigma_box.grid(row=7, column=1, padx=5, pady=5, sticky='w')
        else:
            self.toggle_canvas.create_image(0, 0, anchor='nw', image=self.imgHopfield)
            self.is_hopfield = True
            self.dataDropDown.config(values=self.hopfieldOption)
            self.dim_label.config(text='Pattern Size: ')
            self.dim_text.config(text='-')
            self.sample_num.config(text='-')
            self.train_btn.config(text='Train Hopfield')
            self.map_size_label.grid_forget()
            self.map_size_text.grid_forget()
            self.sigma_label.grid_forget()
            self.sigma_box.grid_forget()


        self.dataDropDown.set('Select a Dataset')

    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            data = file.readlines()
        return [str2float(item.split()) for item in data]

    def load(self, file_name):
        self.file_name = file_name
        if self.is_hopfield:
            self.data = self.read_file(self.get_path('data/hopfield/' + file_name)) 
            if file_name == 'Basic_Training.txt':
                self.dim_text.config(text="9*12")
                self.sample_num.config(text='3')
            else:
                self.dim_text.config(text="10*10")
                self.sample_num.config(text='15')
        else:
            self.data = self.read_file(self.get_path('data/som/' + file_name))
            self.dim = len(self.data[0]) - 1
            self.dim_text.config(text=self.dim)
            self.sample_num.config(text=len(self.data))
            self.map_size_text.config(text='-')


    def get_path(self, folder):
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller uses a temporary folder named _MEIPASS to extract files
            return os.path.join(sys._MEIPASS, f"{folder}")
        else:
            # In development mode
            return os.path.join(os.path.abspath("."), f"{folder}")
        
    def get_all_files_name(self, folder):
        return os.listdir(self.get_path(folder))
    
    def open(self):
        self.container.mainloop()

    def save_graph_frame(self):
        # Get the frame's coordinates
        try:
            filename = asksaveasfilename(initialfile='Screenshot.png', defaultextension=".png", filetypes=[("All Files", "*.*"), ("Portable Graphics Format", "*.png")])
            time.sleep(1)
            if platform.system() == 'Windows' :
                window = pygetwindow.getwindowswithTitle('Neural Network HW3 - Hopfield Network + Self-Organizing Map')[0]
                left, top = window.topleft
                right, bottom = window.bottomright
                pyautogui.screenshot(filename)


            elif platform.system() == 'Darwin':
                x, y, width, height = pygetwindow.getWindowGeometry('Neural Network HW3 - Hopfield Network + Self-Organizing Map')
                image = pyautogui.screenshot(region=(int(x), int(y), int(width), int(height)), imageFilename=filename)
                image.save(filename)
                    
        except Exception as e:
            print(e)

    def train_model(self):
        # print(self.data, self.get_current_epoch(), self.get_current_lrn_rate())
        if self.data is None:
            messagebox.showerror('Error', 'Please select a dataset')
            return
        print('Training...')
        try: 
            if self.is_hopfield:
                model = Hopfield(self.data)
            else:
                self.train_btn.config(state="disabled")
                self.toggle_canvas.unbind("<Button-1>")

                model = SOM(self.dataDropDown.get(), self.data, self.get_current_epoch(), self.get_current_lrn_rate(), self.get_current_sigma())
                self.map_size_text.config(text=str(model.get_map_size()))
                
                def update_canvas(fig, fig2):
                    if self.figure:
                        self.figure.clf()  # Clear the previous figure
                    self.figure = fig
                    self.canvas.figure = self.figure
                    self.canvas.draw()
                    self.canvas.get_tk_widget().update()

                    if self.figure_2:
                        self.figure_2.clf()
                    self.figure_2 = fig2
                    self.canvas_2.figure = self.figure_2
                    self.canvas_2.draw()
                    self.canvas_2.get_tk_widget().update()

                # Train the model with the update callback
                model.train(update_callback=update_canvas)
                self.train_btn.config(state="normal")
                self.toggle_canvas.bind("<Button-1>", lambda _: self.switch())
                
                

        except Exception as e:
            print(e)
            messagebox.showerror('Error', 'Training failed')
            return
    
    def get_current_epoch(self):
        return int(self.epoch_box.get())
    
    def get_current_lrn_rate(self):
        return float(self.lrn_rate_box.get())
    
    def get_current_sigma(self):
        return float(self.sigma_box.get())
