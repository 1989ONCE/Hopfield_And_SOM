import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageGrab
# import pygetwindow
# import pyautogui
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
        self.model = None
        self.epoch = 0
        self.lr = 0.1
        

        # container initialization
        self.container = tk.Tk()
        self.container.config(bg='white')
        self.container.maxsize(app_width, app_height)
        self.container.title(app_name)
        self.container.geometry(str(app_width) + 'x' + str(app_height))


        # components initialization
        self.graph_frame = tk.Frame(self.container, width=810, height=735, bg='white')
        self.setting_frame = tk.Frame(self.container, width=500, height=735, bg='white')

        self.som_canvas = FigureCanvasTkAgg(master = self.graph_frame)
        self.som_canvas.get_tk_widget().config(width=790, height=700)

        self.som_canvas_2 = FigureCanvasTkAgg(master = self.setting_frame)
        self.som_canvas_2.get_tk_widget().config(width=480, height=420)

        self.hopfield_correct_canvas = FigureCanvasTkAgg(master = self.setting_frame)
        self.hopfield_correct_canvas.get_tk_widget().config(width=480, height=420)
        
        self.hopfield_predict_canvas = FigureCanvasTkAgg(master = self.graph_frame)
        self.hopfield_predict_canvas.get_tk_widget().config(width=480, height=420)

        self.is_hopfield = False
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
        self.toggle_canvas.create_image(0, 0, anchor='nw', image=self.imgSOM)
        self.toggle_canvas.bind("<Button-1>", lambda _: self.switch())

        
        self.dataDropDown = ttk.Combobox(master = self.setting_frame,
                        values=self.somOption)
        
        self.dim_label = tk.Label(self.setting_frame, text='Dim:', bg='white')
        self.dim_text = tk.Label(self.setting_frame, text='-', bg='white')
        self.sample_num_label = tk.Label(self.setting_frame, text='Numbers of Samples: ', bg='white')
        self.sample_num = tk.Label(self.setting_frame, text='-', bg='white')
        self.epoch_label = tk.Label(self.setting_frame, text='Epoch:', bg='white')
        self.epoch_box = tk.Spinbox(self.setting_frame, increment=1, from_=0, width=5, bg='white', textvariable=tk.StringVar(value='100'))
        self.lrn_rate_label = tk.Label(self.setting_frame, text='Learning Rate:', bg='white')
        self.lrn_rate_box = tk.Spinbox(self.setting_frame,  format="%.2f", increment=0.01, from_=0.0, to=1.0, width=5, textvariable=tk.StringVar(value='0.8'))
        self.map_size_label = tk.Label(self.setting_frame, text='Map Size:', bg='white')
        self.map_size_text = tk.Label(self.setting_frame, text='-', bg='white')
        self.sigma_label = tk.Label(self.setting_frame, text='Sigma:', bg='white')
        self.sigma_box = tk.Spinbox(self.setting_frame,  format="%.2f", increment=0.1, from_=0.0, width=5, bg='white', textvariable=tk.StringVar(value='3.0'))
        self.train_btn = tk.Button(master = self.setting_frame,  
                     command = self.train_model, 
                     height = 2,  
                     width = 20, 
                     text = "Train SOM",
                     highlightbackground='white') 
        self.recall_btn = tk.Button(master = self.graph_frame,
                        command = self.recall_model,
                        height = 2,
                        width = 10,
                        text = "Start Recall",
                        highlightbackground='white')
        self.prev_btn = tk.Button(master = self.setting_frame,
                        command = self.prev_figure,
                        height = 2,
                        width = 10,
                        text = "Previous",
                        highlightbackground='white')
        self.next_btn = tk.Button(master = self.setting_frame,
                        command = self.next_figure,
                        height = 2,
                        width = 10,
                        text = "Next",
                        highlightbackground='white')
        # self.save_graph_frame_btn = tk.Button(master = self.setting_frame,  
        #              command = self.save_graph_frame, 
        #              height = 2,  
        #              width = 16, 
        #              text = "Take screenshot",
        #              highlightbackground='white')

        # components placing
        self.setting_frame.place(x=10, y=5)
        self.graph_frame.place(x=515, y=5)
        self.som_canvas.get_tk_widget().place(x=10, y=20)
        self.som_canvas_2.get_tk_widget().grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        self.som_figure = None
        self.som_figure_2 = None

        self.hopfield_correct_figure = []
        # self.hopfield_noise_figure = []
        self.hopfield_predict_figure = []
        # self.hopfield_noise_data = None
        self.hopfield_weight = None

        # toggle and data dropdown
        self.toggle_canvas.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.dataDropDown.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.dataDropDown.set('Select a Dataset')
        self.dataDropDown.bind("<<ComboboxSelected>>",  lambda e: self.load(self.dataDropDown.get()))

        self.dim_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.dim_text.grid(row=2, column=1, padx=5, pady=5, sticky='w')   
        self.sample_num_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.sample_num.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.map_size_label.grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.map_size_text.grid(row=4, column=1, padx=5, pady=5, sticky='w')
        self.epoch_label.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.epoch_box.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        self.lrn_rate_label.grid(row=6, column=0, padx=5, pady=5, sticky='w')
        self.lrn_rate_box.grid(row=6, column=1, padx=5, pady=5, sticky='w')
        self.sigma_label.grid(row=7, column=0, padx=5, pady=5, sticky='w')
        self.sigma_box.grid(row=7, column=1, padx=5, pady=5, sticky='w')
        self.train_btn.grid(row=8, column=0, padx=5, pady=5, sticky='w')
        # self.save_graph_frame_btn.grid(row=8, column=1, padx=5, pady=5, sticky='w')
    
    def clear_hopfield_canvas(self):
        new_fig = Figure()
        self.hopfield_correct_canvas.figure = new_fig
        self.hopfield_predict_canvas.figure = new_fig
        self.hopfield_correct_canvas.draw()
        self.hopfield_correct_canvas.get_tk_widget().update()
        self.hopfield_predict_canvas.draw()
        self.hopfield_predict_canvas.get_tk_widget().update()

    def clear_canvas(self):
        if self.som_figure:
            self.som_figure.clf()
            self.som_canvas.draw_idle()
        if self.som_figure_2:
            self.som_figure_2.clf()
            self.som_canvas_2.draw_idle()

    # Determine is hopfield or SOM
    def switch(self):
        self.data = None
        self.clear_canvas()
        self.clear_hopfield_canvas()
        self.train_btn.config(state='normal')

        if self.is_hopfield:
            self.toggle_canvas.create_image(0, 0, anchor='nw', image=self.imgSOM)
            self.is_hopfield = False
            self.dataDropDown.config(values=self.somOption)
            self.sample_num.config(text='-')
            self.dim_label.config(text='Dim:')
            self.dim_text.config(text="-")
            self.train_btn.config(text='Train SOM')
            self.map_size_label.grid(row=4, column=0, padx=5, pady=5, sticky='w')
            self.map_size_text.grid(row=4, column=1, padx=5, pady=5, sticky='w')
            self.map_size_label.config(text='Map Size:')
            self.map_size_text.config(text='-')
            self.sigma_label.grid(row=7, column=0, padx=5, pady=5, sticky='w')
            self.sigma_box.grid(row=7, column=1, padx=5, pady=5, sticky='w')
            self.epoch_label.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            self.epoch_box.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            self.lrn_rate_label.grid(row=6, column=0, padx=5, pady=5, sticky='w')
            self.lrn_rate_label.config(text='Learning Rate:')
            self.lrn_rate_box.config(from_=0.0, to=1.0, textvariable=tk.StringVar(value='0.8'))
            self.som_canvas.get_tk_widget().place(x=10, y=30)
            self.som_canvas_2.get_tk_widget().grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky='w')
            self.hopfield_correct_canvas.get_tk_widget().grid_forget()
            self.hopfield_predict_canvas.get_tk_widget().place_forget()
            self.recall_btn.place_forget()
            self.prev_btn.grid_forget()
            self.next_btn.grid_forget()
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
            self.epoch_label.grid_forget()
            self.epoch_box.grid_forget()
            # self.lrn_rate_label.config(text='noise level for training:')
            # self.lrn_rate_box.config(from_=0.0, to=0.5, textvariable=tk.StringVar(value='0.25'))
            self.lrn_rate_label.grid_forget()
            self.lrn_rate_box.grid_forget()
            
            self.som_canvas.get_tk_widget().place_forget()
            self.som_canvas_2.get_tk_widget().grid_forget()
            self.hopfield_correct_canvas.get_tk_widget().grid(row=9, column=0, columnspan=2)
            self.hopfield_predict_canvas.get_tk_widget().place(x=10, y=155)

        self.dataDropDown.set('Select a Dataset')

    def read_som_file(self, file_path):
        with open(file_path, 'r') as file:
            data = file.readlines()
        return [str2float(item.split()) for item in data]

    def read_hopfield_file(self, file_path):
        with open(file_path, 'r') as file:
            data = file.readlines()
            data_temp = []
            data_list = []
            total_data = []

            self.column = len(data[0]) - 1
            self.row = 0
            for line in data:
                if line == '\n':
                    total_data.append(data_list)
                    data_list = []
                    
                else:
                    self.row += 1
                    for i in range(len(line)):
                        if line[i] == ' ':
                            data_temp.append(-1)
                        elif line[i] == '1':
                            data_temp.append(1)
                        
                        if line[i] == '\n' or i == len(line) - 1:
                            data_list.append(data_temp)
                            data_temp = []
            # 因為最後一個line後面沒有空行，所以要再append一次
            total_data.append(data_list)
            data_list = [] 
            self.row //= len(total_data)
            self.dim = f"{self.column}*{self.row}"
        return total_data
    
    def load(self, file_name):

        self.file_name = file_name
        self.train_btn.config(state='normal')
        self.recall_btn.config(state='normal', text='Start Recall')
        self.recall_btn.place_forget()
        self.prev_btn.config(state='disabled')
        self.next_btn.config(state='normal')
        self.clear_canvas()

        if self.is_hopfield:
            self.prev_btn.grid(row=10, column=0, padx=5, pady=5, sticky='w')
            self.next_btn.grid(row=10, column=1, padx=5, pady=5, sticky='w')
        
            self.train_btn.config(text='Train Hopfield')
            self.hopfield_correct_figure = []
            # self.hopfield_noise_figure = []
            self.hopfield_predict_figure = []
            self.hopfield_weight = None
            # self.hopfield_noise_data = None

            self.data = self.read_hopfield_file(self.get_path('data/hopfield/' + file_name)) 
            
            self.model = Hopfield()
            self.model.setData(self.data)

            if file_name == 'Basic_Training.txt':   
                self.test_data = self.read_hopfield_file(self.get_path('data/hopfield/Basic_Testing.txt'))
            else:
                self.test_data = self.read_hopfield_file(self.get_path('data/hopfield/Bonus_Testing.txt'))

            self.dim_text.config(text=self.dim)
            self.sample_num.config(text=len(self.data))
            
            for i in range(len(self.data)):
                pattern = self.data[i]
                fig = self.draw_patterns(pattern, i, 'Pattern Answer')
                self.hopfield_correct_figure.append(fig)
            
            for i in range(len(self.test_data)):
                pattern = self.test_data[i]
                fig = self.draw_patterns(pattern, i, 'Testing Dataset')
                self.hopfield_predict_figure.append(fig)
            
            self.hopfield_correct_canvas.figure = self.hopfield_correct_figure[0]
            self.hopfield_correct_canvas.draw()
            self.hopfield_correct_canvas.get_tk_widget().update()

            self.hopfield_predict_canvas.figure = self.hopfield_predict_figure[0]
            self.hopfield_predict_canvas.draw()
            self.hopfield_predict_canvas.get_tk_widget().update()
        else:
            self.data = self.read_som_file(self.get_path('data/som/' + file_name))
            self.dim = len(self.data[0]) - 1
            self.dim_text.config(text=self.dim)
            self.sample_num.config(text=len(self.data))
            self.map_size_text.config(text='-')

    def draw_patterns(self, data, index, type_name):
        fig, ax = plt.subplots(figsize=(4, 4))
        if index is not None:
            ax.set_title(f'{type_name}: Pattern {index+1}')
        else:
            ax.set_title(f'{type_name}')
        
        # 設置 x 軸和 y 軸的刻度為空
        ax.set_xticks([])
        ax.set_yticks([])

        # 設置正方形的格子
        ax.set_aspect('equal')  

        # 反轉 y 軸，讓圖像顯示正確的方向
        for i in range(self.column):
            for j in range(self.row):
                if data[j][i] == 1:
                    ax.add_patch(plt.Rectangle((i, self.row - j - 1), 1, 1, fill=True, color='black'))
                else:
                    ax.add_patch(plt.Rectangle((i, self.row - j - 1), 1, 1, fill=True, color='white'))

        # 繪製背景格線
        for i in range(self.column):
            ax.axvline(i, color='gray', lw=0.5)
        for i in range(self.row + 1):
            ax.axhline(i, color='gray', lw=0.5)

        # 移除多餘的邊框，填滿整個區域
        ax.set_xlim(0, self.column)
        ax.set_ylim(0, self.row)

        # plt.tight_layout(pad=0)  # 縮小圖形的邊距
        
        plt.close() # 關閉圖形，避免佔用記憶體
        return fig

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

    # def save_graph_frame(self):
    #     # Get the frame's coordinates
    #     try:
    #         filename = asksaveasfilename(initialfile='Screenshot.png', defaultextension=".png", filetypes=[("All Files", "*.*"), ("Portable Graphics Format", "*.png")])
    #         time.sleep(1)
    #         if platform.system() == 'Windows' :
    #             window = pygetwindow.getwindowswithTitle('Neural Network HW3 - Hopfield Network + Self-Organizing Map')[0]
    #             left, top = window.topleft
    #             right, bottom = window.bottomright
    #             pyautogui.screenshot(filename)


    #         elif platform.system() == 'Darwin':
    #             x, y, width, height = pygetwindow.getWindowGeometry('Neural Network HW3 - Hopfield Network + Self-Organizing Map')
    #             image = pyautogui.screenshot(region=(int(x), int(y), int(width), int(height)), imageFilename=filename)
    #             image.save(filename)
                    
    #     except Exception as e:
    #         print(e)

    def train_model(self):
        # print(self.data, self.get_current_epoch(), self.get_current_lrn_rate())
        if self.data is None:
            messagebox.showerror('Error', 'Please select a dataset')
            return
        print('Training...')
        try: 
            if self.is_hopfield:
                self.recall_btn.place(x=20, y=100)
                # if float(self.lrn_rate_box.get()) < 0.0 or float(self.lrn_rate_box.get()) > 0.5:
                #     messagebox.showerror('Error', 'Invalid noise level')
                #     return
                self.train_btn.config(state="disabled")
                self.toggle_canvas.unbind("<Button-1>")
                self.hopfield_weight = self.model.train()
                if self.hopfield_weight.any:
                    self.train_btn.config(text='Finish Training')
                self.toggle_canvas.bind("<Button-1>", lambda _: self.switch())
                
            else:
                if self.lrn_rate_box.get() == '0.0' or self.sigma_box.get() == '1.0' or float(self.lrn_rate_box.get()) > 1.0:
                    messagebox.showerror('Error', 'Invalid Learning rate or sigma')
                    return
                self.train_btn.config(state="disabled")
                self.toggle_canvas.unbind("<Button-1>")

                self.model = SOM(self.dataDropDown.get(), self.data, self.get_current_epoch(), self.get_current_lrn_rate(), self.get_current_sigma())
                self.map_size_text.config(text=str(self.model.get_map_size()))
                
                def update_canvas(fig, fig2):
                    if self.som_figure:
                        self.som_figure.clf()  # Clear the previous figure
                    self.som_figure = fig
                    self.som_canvas.figure = self.som_figure
                    self.som_canvas.draw_idle()
                    self.som_canvas.get_tk_widget().update()

                    if self.som_figure_2:
                        self.som_figure_2.clf()
                    self.som_figure_2 = fig2
                    self.som_canvas_2.figure = self.som_figure_2
                    self.som_canvas_2.draw_idle()
                    self.som_canvas_2.get_tk_widget().update()

                # Train the model with the update callback
                self.model.train(update_callback=update_canvas)
                self.train_btn.config(state="normal")
                self.toggle_canvas.bind("<Button-1>", lambda _: self.switch())
        except Exception as e:
            print(e)
            messagebox.showerror('Error', 'Training failed')
            return
        
    def recall_model(self):
        if self.data is None:
            messagebox.showerror('Error', 'Please select a dataset')
            return
        
        if self.hopfield_weight is None:
            messagebox.showerror('Error', 'Please train the model first')
            return
        try:
            correct_index = 0
            def update_hopfield_canvas(iter, new_pattern):
                
                predict_figure = self.draw_patterns(new_pattern, None, 'Recalled Pattern - Iteration ' + str(iter))
                self.hopfield_predict_canvas.figure = predict_figure
                self.hopfield_predict_canvas.draw_idle()
                self.hopfield_predict_canvas.get_tk_widget().update()
            print('Recalling...')
            self.recall_btn.config(state="disabled", text="Recalling...")

            # Train the model with the update callback
            for i in range(len(self.hopfield_predict_figure)):
                if self.hopfield_predict_canvas.figure == self.hopfield_predict_figure[i]:
                    self.predict_data = self.test_data[i]
                    correct_index = i
            
            new_pattern_data = self.model.recall(self.predict_data, 10, update_callback=update_hopfield_canvas)
            
            print('Recall completed')

            if np.array_equal(new_pattern_data, self.data[correct_index]):
                self.recall_btn.config(state='disabled', text="Recall Successfully")
                messagebox.showinfo('Recall', 'Recall Successfully')
            else:
                self.recall_btn.config(state='disabled', text="Recall Failed")
                messagebox.showinfo('Recall', 'Recall Failed')
                
            self.toggle_canvas.bind("<Button-1>", lambda _: self.switch())
            
                
        except Exception as e:
            print(e)
            messagebox.showerror('Error', 'Recall failed')
            return
       
    def get_current_epoch(self):
        return int(self.epoch_box.get())
    
    def get_current_lrn_rate(self):
        return float(self.lrn_rate_box.get())
    
    def get_current_sigma(self):
        return float(self.sigma_box.get())

    def prev_figure(self):
        self.next_btn.config(state='normal')
        self.recall_btn.config(state='normal', text='Start Recall')

        for i in range(len(self.hopfield_correct_figure)):
            if self.hopfield_correct_canvas.figure == self.hopfield_correct_figure[i]:
                self.clear_hopfield_canvas()
                self.hopfield_correct_canvas.figure = self.hopfield_correct_figure[i-1]
                self.hopfield_predict_canvas.figure = self.hopfield_predict_figure[i-1]
                if i-1 == 0 or i == 0:
                    self.prev_btn.config(state='disabled')
                self.hopfield_correct_canvas.draw()
                self.hopfield_correct_canvas.get_tk_widget().update()
                   
                self.hopfield_predict_canvas.draw()
                self.hopfield_predict_canvas.get_tk_widget().update()
                break

    def next_figure(self):
        try:
            self.prev_btn.config(state='normal')
            self.recall_btn.config(state='normal', text='Start Recall')
        except:
            print('error')

        for i in range(len(self.hopfield_correct_figure)):
            if self.hopfield_correct_canvas.figure == self.hopfield_correct_figure[i]:
                self.clear_hopfield_canvas()
                self.hopfield_correct_canvas.figure = self.hopfield_correct_figure[i+1]
                self.hopfield_predict_canvas.figure = self.hopfield_predict_figure[i+1]
                if i+1 == len(self.hopfield_correct_figure) - 1 or i == len(self.hopfield_correct_figure) - 1:
                    self.next_btn.config(state='disabled')
                
                self.hopfield_correct_canvas.draw()
                self.hopfield_correct_canvas.get_tk_widget().update()
               
                self.hopfield_predict_canvas.draw()
                self.hopfield_predict_canvas.get_tk_widget().update()
                break
    def add_noise(self):
        if self.data is None:
            messagebox.showerror('Error', 'Please select a dataset')
            return
        # noise_level = float(self.lrn_rate_box.get())
        # if noise_level < 0.0 or noise_level > 0.5:
        #     messagebox.showerror('Error', 'Invalid noise level')
        #     return
        
        try:
            pass
            # noise_level = float(self.lrn_rate_box.get())
            # if noise_level < 0.0 or noise_level > 0.5:
            #     messagebox.showerror('Error', 'Invalid noise level')
            #     return

        
            # noisy_data = self.model.add_noise(self.data, noise_level)

            # for i in range(len(noisy_data)):
            #     fig = self.draw_patterns(noisy_data[i], i, 'Training Dataset After Adding Noise')
            #     self.hopfield_noise_figure.append(fig)
            # self.hopfield_noise_data = noisy_data
        except Exception as e:
            print(e)
            messagebox.showerror('Error', 'Add noise failed')
            return
