from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os


class ColWin(QDialog):
    def __init__(self, SpeAn, signal, t_pca, p_pca, t_der1, p_der1, t_der2, p_der2, waves):
        super(ColWin, self).__init__()
        print(os.path.dirname(os.path.abspath(__file__)))
        loadUi("C:\\PCA_with_R\\ColumnWindow.ui", self)
        self.SpeAn = SpeAn
        self.signal = signal
        self.t_matrix_pca = t_pca
        self.p_matrix_pca = p_pca
        self.t_matrix_der1 = t_der1
        self.p_matrix_der1 = p_der1
        self.t_matrix_der2 = t_der2
        self.p_matrix_der2 = p_der2
        self.waves = waves
        self.colGraph = MplWidget(signal, self.colGraph)
        self.pushButton.clicked.connect(self.showGraphColumn)
        self.CloseButton.clicked.connect(self.close)

    def showGraphColumn(self):
        self.radioButtonChecking()
        self.filenames = self.SpeAn.filenames
        if self.Columns_int.toPlainText() != '':
            Columns_temp = self.Columns_int.toPlainText()
        else:
            Columns_temp = "1, 2, 3"
        Columns_temp = Columns_temp.replace(' ', '')
        Columns_temp = Columns_temp.replace('.', ',')
        input_temp = Columns_temp.split(',')
        self.Columns = []
        for j in input_temp:
            self.Columns.append(int(j))
        if self.Columns == []:
            pass
        else:
            if self.signal == 1:
                self.plotScores()
            elif self.signal == 2:
                self.plotLoadings()
            elif self.signal == 3:
                self.plot3D()

    def radioButtonChecking(self):
        if self.SpectraButton.isChecked():
            self.t_matrix = self.t_matrix_pca
            self.p_matrix = self.p_matrix_pca
            self.waves_for_graph = self.waves[0]
        elif self.Derivative_1_Button.isChecked():
            self.t_matrix = self.t_matrix_der1
            self.p_matrix = self.p_matrix_der1
            self.waves_for_graph = self.waves[1]
        elif self.Derivative_2_Button.isChecked():
            self.t_matrix = self.t_matrix_der2
            self.p_matrix = self.p_matrix_der2
            self.waves_for_graph = self.waves[2]

    def plotScores(self):
        self.colGraph.canvas.ax.clear()
        first_column = self.Columns[0]
        second_column = self.Columns[1]
        first_column -= 1
        second_column -= 1
        self.x = self.t_matrix[:, first_column]
        self.y = self.t_matrix[:, second_column]
        for index in range(len(self.filenames)):
            if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="red", marker="o", s=50)
                # self.colGraph.canvas.ax.annotate(self.filenames[index], (self.x[index], self.y[index]))
            elif self.filenames[index][0] == 'N':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="blue", marker="P", s=50)
                # self.colGraph.canvas.ax.annotate(self.filenames[index], (self.x[index], self.y[index]))
            elif self.filenames[index][0] == 'D':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="green", marker="*", s=50)
                # self.colGraph.canvas.ax.annotate(self.filenames[index], (self.x[index], self.y[index]))
            elif (self.filenames[index][0] == 'O') or (self.filenames[index][0] == 'B'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="black", marker="*", s=50)
                # plt.annotate(filenames[index], (x[index], y[index]))
        self.colGraph.canvas.draw()


    def plotLoadings(self):
        first_column = self.Columns[0]
        second_column = self.Columns[1]
        first_column -= 1
        second_column -= 1

        figs = plt.figure()
        axs = figs.add_subplot(111)
        ys = list(self.p_matrix[:, first_column])
        xs = self.waves_for_graph
        axs.scatter(xs, ys, color="black", marker="o", s=12)
        plt.show()

        self.x = self.p_matrix[:, first_column]
        self.y = self.p_matrix[:, second_column]
        self.colGraph.canvas.ax.clear()
        self.colGraph.canvas.ax.scatter(self.x, self.y, color="black", marker="o", s=12)
        plt.axvline(x=1652.5, color='red', label='Alpha-helices', linewidth=3.5, alpha=0.5)
        plt.axvline(x=1629.5, color='blue', label='Beta-sheets', linewidth=11.5, alpha=0.5)
        plt.axvline(x=1682.5, color='blue', label='Beta-sheets', linewidth=12.5, alpha=0.5)
        plt.axvline(x=1631, color='blue', label='Beta-sheets', linewidth=1, alpha=0.5)
        plt.axvline(x=1664, color='green', label='Beta-turns', linewidth=1, alpha=0.5)
        plt.axvline(x=1672, color='green', label='Beta-turns', linewidth=1, alpha=0.5)
        plt.axvline(x=1684, color='green', label='Beta-turns', linewidth=1, alpha=0.5)
        plt.axvline(x=1690, color='green', label='Beta-turns', linewidth=1, alpha=0.5)
        plt.axvline(x=1647, color='orange', label='Random-coil', linewidth=2, alpha=0.5)
        self.colGraph.canvas.draw()


    def plot3D(self):
        self.colGraph.canvas.ax.clear()
        first_column = self.Columns[0]
        second_column = self.Columns[1]
        third_column = self.Columns[2]
        first_column -= 1
        second_column -= 1
        third_column -= 1
        self.x = self.t_matrix[:, first_column]
        self.y = self.t_matrix[:, second_column]
        self.z = self.t_matrix[:, third_column]
        for index in range(len(self.filenames)):
            if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="red")
            elif self.filenames[index][0] == 'N':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="blue")
            elif self.filenames[index][0] == 'D':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="green")
            elif (self.filenames[index][0] == 'O') or (self.filenames[index][0] == 'B'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="black")
        self.colGraph.canvas.draw()


class MplCanvas(Canvas):
    def __init__(self, signal):
        dpi = 100
        self.fig = Figure(figsize=(1240/dpi, 600/dpi), dpi=dpi)
        if signal == 1:
            self.ax = self.fig.add_subplot(111)
        elif signal == 2:
            self.ax = self.fig.add_subplot(111)
        elif signal == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            label.set_fontsize(16)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class MplWidget(QtWidgets.QWidget):
    def __init__(self, signal, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas(signal)
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
