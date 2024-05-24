import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import copy
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QMessageBox, QWidget,QVBoxLayout
from PyQt5.QtGui import QPixmap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QTimer
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
# from train import createSequence,split_dataset

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# 导入matplotlib模块并使用Qt5Agg
import matplotlib

matplotlib.use('Qt5Agg')
# 使用 matplotlib中的FigureCanvas (在使用 Qt5 Backends中 FigureCanvas继承自QtWidgets.QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import *
import threading
seed = 3407

def split_dataset(df, window_size):
    seq, label = createSequence(df, window_size)

    # train
    train_seq, test_seq = train_test_split(seq, test_size=0.4, random_state=seed)
    train_label, test_label = train_test_split(label, test_size=0.4, random_state=seed)

    # val & test
    val_seq, test_seq = train_test_split(test_seq, test_size=0.5, random_state=seed)
    val_label, test_label = train_test_split(test_label, test_size=0.5, random_state=seed)

    return train_seq, train_label, val_seq, val_label, test_seq, test_label
def createSequence(df, window_size):
    # 找到轨迹长度 >= 15的所有轨迹
    # 按照滑动窗口进行划分


    seq = []
    label = []
    df = df.drop(columns = 'Unnamed: 0')
    df = df.loc[~(df == 0).all(axis=1)]
    df = df.rolling(window=5).mean().bfill()  #平滑处理
    trajectory = df.values.tolist()
    traj = trajectory
    x = []
    for tra in traj:
        sum = 0
        for tr in tra:
            sum+=tr
        if sum!=0:
            for i in range(len(tra)):
                if tra[i] == 0:
                    if i == 0:
                        tra[i] = tra[i+1]
                    elif i == len(tra) - 1:
                        tra[i] = tra[i-1]
                    else:
                        tra[i] = (tra[i-1]+tra[i+1])/2
            x.append(tra)
    trajectory = x

    num_splits = len(trajectory) - window_size + 1

    for i in range(num_splits):
        seq.append(trajectory[i:i + window_size - 1])
        label.append(trajectory[i + window_size - 1])

    seq = torch.tensor(np.array(seq), dtype=torch.float32).to(device)
    label = torch.tensor(np.array(label), dtype=torch.float32).to(device)
    return seq, label

class BiLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Bi-LSTM 需要两个隐藏状态
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out)
        out = self.fc(out[:, -1, :])
        return out

class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

	# 导入网络结构

class running(threading.Thread):
    def __init__(self):
        super().__init__()
        df = pd.read_csv(
            'data\METR-LA.csv')
        self.output = 0
        window_size = 10
        # df = df.drop(columns='Unnamed: 0')
        # df = df.loc[~(df == 0).all(axis=1)]
        # df = df.rolling(window=5).mean().bfill()  # 平滑处理
        # self.df = df
        # trajectory = df.values.tolist()
        # traj = trajectory
        # x = []
        # for tra in traj:
        #     sum = 0
        #     for tr in tra:
        #         sum += tr
        #     if sum != 0:
        #         for i in range(len(tra)):
        #             if tra[i] == 0:
        #                 if i == 0:
        #                     tra[i] = tra[i + 1]
        #                 elif i == len(tra) - 1:
        #                     tra[i] = tra[i - 1]
        #                 else:
        #                     tra[i] = (tra[i - 1] + tra[i + 1]) / 2
        #         x.append(tra)
        # trajectory = x
        # self.seq = torch.tensor(np.array(trajectory[:-1]), dtype=torch.float32).to(device)

        train_seq, train_label, val_seq, val_label, test_seq, test_label = split_dataset(df, window_size)
        test_set = TensorDataset(test_seq, test_label)
        self.test_loader = DataLoader(test_set, batch_size=32, shuffle=False, generator=torch.Generator(device=device))
        self.model = GRUPredictor(207,108,207)
        # self.model = BiLSTMPredictor(207,108,207)
        self.model.load_state_dict(torch.load(
            '1715064525lr_0.0001ws_10epoch_300\GRU.pt',map_location=torch.device('cuda:0')))
        self.input = test_seq

        self._stop_event = threading.Event()
        self.predictions = []
    def stop(self):
        self._stop_event.set()
    def run(self):
        while not self._stop_event.is_set():
            with torch.no_grad():
                # self.output = self.model(self.input)
                # self.seq = torch.cat((self.seq,self.output),dim = 0)
                # self.input = torch.cat((self.input[:,1:,:],self.output.unsqueeze(0)),dim = 1)
                for data in self.test_loader:
                    self.input, self.labels = data
                    self.output = self.model(self.input)
                    self.predictions.append(self.output)


device = 'cuda:0'
torch.cuda.set_device(device)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

class MyMatplotlibFigure(FigureCanvasQTAgg):
    def __init__(self,width = 10,height = 10,dpi = 100):
        # plt.rcParams['figure.facecolor'] = 'r'  # 设置窗体颜色
        # plt.rcParams['axes.facecolor'] = 'w'  # 设置绘图区颜色
        # 创建一个Figure,该Figure为matplotlib下的Figure，不是matplotlib.pyplot下面的Figure
        self.figs = Figure(figsize=(width, height), dpi=dpi)
        super(MyMatplotlibFigure, self).__init__(self.figs)  # 在父类种激活self.fig，
        self.axes = self.figs.add_subplot(111)  # 添加绘图区

    def mat_plot_drow_axes(self, t, s):
        """
        用清除画布刷新的方法绘图
        :return:
        """
        self.figs.clf()  # 清除绘图区
        self.axes = self.figs.add_subplot(111)
        self.axes.spines['top'].set_visible(False)  # 顶边界不可见
        self.axes.spines['right'].set_visible(False)  # 右边界不可见
        # 设置左、下边界在（0，0）处相交
        # self.axes.spines['bottom'].set_position(('data', 0))  # 设置y轴线原点数据为 0
        self.axes.spines['left'].set_position(('data', 0))  # 设置x轴线原点数据为 0
        self.axes.plot(t, s, 'b', linewidth=0.5)
        self.figs.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.figs.canvas.flush_events()  # 画布刷新self.figs.canvas





class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.generator = running()
        self.generator.start()
        self.flag3 = 0

        df = pd.read_csv(
            'data\METR-LA.csv')[-11:]

        df = df.drop(columns='Unnamed: 0')
        df = df.loc[~(df == 0).all(axis=1)]
        df = df.rolling(window=5).mean().bfill()  # 平滑处理
        self.df = df
        trajectory = df.values.tolist()
        traj = trajectory
        x = []
        for tra in traj:
            sum = 0
            for tr in tra:
                sum += tr
            if sum != 0:
                for i in range(len(tra)):
                    if tra[i] == 0:
                        if i == 0:
                            tra[i] = tra[i + 1]
                        elif i == len(tra) - 1:
                            tra[i] = tra[i - 1]
                        else:
                            tra[i] = (tra[i - 1] + tra[i + 1]) / 2
                x.append(tra)
        trajectory = x
        self.seq = torch.tensor(np.array(trajectory), dtype=torch.float32).to(device)
        self.model = torch.load('1715064525lr_0.0001ws_10epoch_300\BiLSTMPredictor.pt')

        self.setWindowTitle("Data Visualization")
        self.setGeometry(400, 300, 1350, 800)

        # Load background image
        self.background = QLabel(self)
        self.background.setPixmap(QPixmap("background_image.png"))
        self.background.setGeometry(0, 0, 1600, 1200)

        # Entry widget for ID input
        self.id_entry = QLineEdit(self)
        self.id_entry.setGeometry(450, 600, 200,100)

        # Button to show line plot
        self.show_plot_button = QPushButton("显示", self)

        self.show_plot_button.setGeometry(675, 600, 200, 100)
        self.show_plot_button.clicked.connect(self.show_line_plot)

        # Sample data
        self.x = np.arange(1, 208)
        self.y = trajectory[-1]

        self.label = QtWidgets.QLabel(self)
        # quanju weizhi daxiao
        self.label.setGeometry(QtCore.QRect(50, 50, 600, 400))

        self.canvas = MyMatplotlibFigure(width = 6,height = 4,dpi =100)

        self.canvas.mat_plot_drow_axes(self.x,self.y)
        self.canvas.figs.suptitle("Global real-time traffic forecasting")

        # self.plot_button = QPushButton("Plot Data", self)
        # self.plot_button.setGeometry(50, 50, 100, 30)
        # self.plot_button.clicked.connect(self.plot_data)
        self.hboxlayout = QtWidgets.QHBoxLayout(self.label)
        self.hboxlayout.addWidget(self.canvas)
        # self.hboxlayout.removeWidget(self.canvas)

        self.flag1 = 0
        self.flag2 = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.plot_data)
        self.timer.start(3000)

        self.label2 = QtWidgets.QLabel(self)
        # dan dian
        self.label2.setGeometry(QtCore.QRect(700, 50, 600, 400))
        self.canvas2 = MyMatplotlibFigure(width = 6,height = 4,dpi =100)
        self.hboxlayout2 = QtWidgets.QHBoxLayout(self.label2)
        self.canvas2.figs.suptitle("Local real-time traffic forecasting")

        # Button to plot data
    def plot_data(self):
        if (self.flag2+1) % 32 == 0:
            self.flag1 += 1
            self.flag2 = 0
        else:
            self.flag2 += 1
        self.hboxlayout.removeWidget(self.canvas)
        #self.y = self.generator.output[0].to('cpu')
        self.y = self.generator.predictions[self.flag1][self.flag2].to('cpu')
        self.canvas = MyMatplotlibFigure(width=6, height=4, dpi=100)
        self.canvas.mat_plot_drow_axes(self.x,self.y.squeeze(0).numpy().tolist())
        # self.canvas.figs.suptitle("Global Real-time Traffic")
        self.hboxlayout.addWidget(self.canvas)

    def show_line_plot(self):
        try:
            id = int(self.id_entry.text())
            if id < 1 or id > 207:
                raise ValueError
            # data = np.random.randint(1, 100, size=id)
            # fig, ax = plt.subplots()
            # ax.plot(range(1, id+1), data)
            # ax.set_xlabel('Time')
            # ax.set_ylabel('Data')
            # ax.set_title(f'Line Plot for ID {id}')
            # canvas = FigureCanvas(fig)
            # self.set CentralWidget(canvas)
            x = []
            data = []
            self.flag3 += 1
            get = self.generator.predictions[self.flag3].to('cpu')
            get = get.numpy().tolist()
            # for i in range(len(get)):
            #     if i % 300 == 0:
            #         x.append(flag)
            #         flag+=1
            #         data.append(get[i][id-1])
            for i in range(len(get)):
                x.append(i)
                data.append(get[i][id-1])
            self.canvas2.mat_plot_drow_axes(x,data)
            self.canvas2.figs.suptitle("Local")
            self.hboxlayout2.addWidget(self.canvas2)

        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid ID. Please enter a number between 1 and 207.")

    def stopGenerator(self):
        if self.generator:
            self.generator.stop()
            self.generator.join()


    def closeEvent(self, event):
        self.stopGenerator()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())




# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         df = pd.read_csv(
#             'G:\北京理工大学学习课程\学期二\数据挖掘\Beijing-Traffic-Track-Data-Mining-master\Beijing-Traffic-Track-Data-Mining-master\METR-LA.csv')[:-9]
#         df = df.drop(columns='Unnamed: 0')
#         df = df.loc[~(df == 0).all(axis=1)]
#         df = df.rolling(window=5).mean().bfill()  # 平滑处理
#         trajectory = df.values.tolist()
#         traj = copy.deepcopy(trajectory)
#         print(len(trajectory))
#         x = []
#         for tra in traj:
#             sum = 0
#             for tr in tra:
#                 sum += tr
#             if sum != 0:
#                 for i in range(len(tra)):
#                     if tra[i] == 0:
#                         if i == 0:
#                             tra[i] = tra[i + 1]
#                         elif i == len(tra) - 1:
#                             tra[i] = tra[i - 1]
#                         else:
#                             tra[i] = (tra[i - 1] + tra[i + 1]) / 2
#                 x.append(tra)
#         trajectory = x
#         self.setWindowTitle("Data Visualization")
#         self.setGeometry(100, 100, 800, 600)
#
#         # Load background image
#         self.background = QLabel(self)
#         self.background.setPixmap(QPixmap("background_image.png"))
#         self.background.setGeometry(0, 0, 800, 600)
#
#         # Sample data
#         self.x = np.arange(1, 208)
#         self.y = trajectory
#
#         # Button to plot data
#         self.plot_button = QPushButton("Plot Data", self)
#         self.plot_button.setGeometry(50, 50, 100, 30)
#         self.plot_button.clicked.connect(self.plot_data)
#
#         # Entry widget for ID input
#         self.id_entry = QLineEdit(self)
#         self.id_entry.setGeometry(50, 100, 100, 30)
#
#         # Button to start updating plot
#         self.start_button = QPushButton("Start", self)
#         self.start_button.setGeometry(50, 130, 100, 30)
#         self.start_button.clicked.connect(self.start_update)
#
#         # Button to stop updating plot
#         self.stop_button = QPushButton("Stop", self)
#         self.stop_button.setGeometry(160, 130, 100, 30)
#         self.stop_button.clicked.connect(self.stop_update)
#
#         # Timer for updating plot
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_plot)
#         self.timer_interval = 1000  # Update every 1 second
#
#         # Create matplotlib figure and canvas
#         self.fig, self.ax = plt.subplots()
#         self.canvas = FigureCanvas(self.fig)
#         self.setCentralWidget(self.canvas)
#
#         self.update_plot_flag = False
#
#     def plot_data(self):
#         self.ax.clear()
#         self.ax.plot(self.x, self.y)
#         self.ax.set_xlabel('X Label')
#         self.ax.set_ylabel('Y Label')
#         self.ax.set_title('Data Plot')
#         self.canvas.draw()
#
#     def show_line_plot(self, id):
#         data = np.random.randint(1, 100, size=id)
#         self.ax.clear()
#         self.ax.plot(range(1, id+1), data)
#         self.ax.set_xlabel('Time')
#         self.ax.set_ylabel('Data')
#         self.ax.set_title(f'Line Plot for ID {id}')
#         self.canvas.draw()
#
#     def start_update(self):
#         self.update_plot_flag = True
#         self.timer.start(self.timer_interval)
#
#     def stop_update(self):
#         self.update_plot_flag = False
#         self.timer.stop()
#
#     def update_plot(self):
#         try:
#             id = int(self.id_entry.text())
#             if id < 1 or id > 207:
#                 raise ValueError
#             self.show_line_plot(id)
#         except ValueError:
#             QMessageBox.critical(self, "Error", "Invalid ID. Please enter a number between 1 and 207.")
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())

