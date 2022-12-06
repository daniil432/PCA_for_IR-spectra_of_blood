import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ColWin(QMainWindow):
    def __init__(self, parent, signal, filenames, t_pca, p_pca, t_der1, p_der1, t_der2, p_der2, tr_pca, pr_pca, waves,):
        super(ColWin, self).__init__(parent)
        loadUi("C:\\PCA_with_R\\interface\\ColumnWindow.ui", self)
        self.parent = parent
        self.t_matrix = None
        self.p_matrix = None
        self.waves_for_graph = None
        self.Columns = []
        self.signal = signal
        self.filenames = filenames
        self.t_matrix_pca = t_pca
        self.p_matrix_pca = p_pca
        self.t_matrix_der1 = t_der1
        self.p_matrix_der1 = p_der1
        self.t_matrix_der2 = t_der2
        self.p_matrix_der2 = p_der2
        self.tr_pca = tr_pca
        self.pr_pca = pr_pca
        self.waves = waves

        self.Columns_int.setPlaceholderText("1,2,3")
        self.pushButton.clicked.connect(self.showGraphColumn)
        self.colGraph = MplWidget(self.signal, self.colGraph)
        self.CloseButton.clicked.connect(self.closeEvent)
        QAction("Quit", self).triggered.connect(self.closeEvent)
        if signal == 2:
            self.label_3.setText('Введите номер столбцa:')
        self.Columns_int.returnPressed.connect(self.pushButton.click)

    def closeEvent(self, event):
        self.parent.show()
        self.close()

    def showGraphColumn(self):
        self.Columns = []
        Columns_temp = self.Columns_int.text()
        Columns_temp = Columns_temp.replace(' ', '')
        Columns_temp = Columns_temp.replace('.', ',')
        Columns_temp = Columns_temp.split(',')
        for j in Columns_temp:
            self.Columns.append(int(j))
        self.radioButtonChecking()
        try:
            if self.signal == 1:
                self.plotScores()
            elif self.signal == 2:
                self.plotLoadings()
            elif self.signal == 3:
                self.plot3D()
        except Exception as error:
            print(error)
            pass

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
        elif self.RatioWave_button.isChecked():
            self.t_matrix = self.tr_pca
            self.p_matrix = self.pr_pca
            self.waves_for_graph = self.p_matrix[:, self.Columns[1]-1]

    def plotScores(self):
        self.colGraph.canvas.ax.clear()
        first_column = self.Columns[0]
        second_column = self.Columns[1]
        first_column -= 1
        second_column -= 1
        x = self.t_matrix[:, first_column]
        y = self.t_matrix[:, second_column]
        for index in range(len(self.filenames)):
            if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                self.colGraph.canvas.ax.scatter(x[index], y[index], color="red", marker="o", s=50)
                # self.colGraph.canvas.ax.annotate(self.filenames[index], (self.x[index], self.y[index]))
            elif self.filenames[index][0] == 'N':
                self.colGraph.canvas.ax.scatter(x[index], y[index], color="blue", marker="P", s=50)
                # self.colGraph.canvas.ax.annotate(self.filenames[index], (self.x[index], self.y[index]))
            elif self.filenames[index][0] == 'D':
                self.colGraph.canvas.ax.scatter(x[index], y[index], color="green", marker="*", s=50)
                # self.colGraph.canvas.ax.annotate(self.filenames[index], (self.x[index], self.y[index]))
            elif (self.filenames[index][0] == 'O') or (self.filenames[index][0] == 'B'):
                self.colGraph.canvas.ax.scatter(x[index], y[index], color="black", marker="*", s=50)
                # plt.annotate(filenames[index], (x[index], y[index]))
        self.colGraph.canvas.draw()

    def plotLoadings(self):
        first_column = self.Columns[0]
        second_column = self.waves_for_graph
        first_column -= 1

        xs = second_column
        ys = list(self.p_matrix[:, first_column])
        self.colGraph.canvas.ax.clear()
        self.colGraph.canvas.ax.scatter(xs, ys, color="black", marker="o", s=12)
        if self.RatioWave_button.isChecked():
            labels = ['M$_{I}$/N$_{1}$', 'M$_{I}$/M$_{S}$', 'M$_{I}$/N$_{2}$', 'M$_{I}$/M$_{T}$', 'M$_{I}$/N$_{3}$',
                      "M$_{I}$/M$_{II}$",
                      'M$_{S}$/N$_{1}$', 'N$_{1}$/N$_{2}$', 'N$_{1}$/M$_{T}$', 'N$_{1}$/N$_{3}$', "M$_{II}$/N$_{1}$",
                      'M$_{S}$/N$_{2}$', 'M$_{S}$/M$_{T}$', 'M$_{S}$/N$_{3}$', "M$_{II}$/M$_{S}$",
                      'M$_{T}$/N$_{2}$', 'N$_{2}$/N$_{3}$', "M$_{II}$/N$_{2}$",
                      'M$_{T}$/N$_{3}$', "M$_{II}$/M$_{T}$",
                      "M$_{II}$/N$_{3}$", 'M$_{I}$', 'N$_{1}$', 'M$_{S}$', 'N$_{2}$', 'M$_{T}$', 'N$_{3}$', "M$_{II}$"]
            for i, txt in enumerate(labels):
                self.colGraph.canvas.ax.annotate(txt, (xs[i], ys[i]))
        if max(second_column) >= 1652.5 and min(second_column) <= 1629.5:
            self.colGraph.canvas.ax.axvline(x=1690, color='green', label='Beta-turns', linewidth=1, alpha=0.5)
            self.colGraph.canvas.ax.axvline(x=1684, color='green', label='Beta-turns', linewidth=1, alpha=0.5)
            self.colGraph.canvas.ax.axvline(x=1682.5, color='blue', label='Beta-sheets', linewidth=12.5, alpha=0.5)
            self.colGraph.canvas.ax.axvline(x=1672, color='green', label='Beta-turns', linewidth=1, alpha=0.5)
            self.colGraph.canvas.ax.axvline(x=1664, color='green', label='Beta-turns', linewidth=1, alpha=0.5)
            self.colGraph.canvas.ax.axvline(x=1652.5, color='red', label='Alpha-helices', linewidth=3.5, alpha=0.5)
            self.colGraph.canvas.ax.axvline(x=1647, color='orange', label='Random-coil', linewidth=2, alpha=0.5)
            self.colGraph.canvas.ax.axvline(x=1631, color='blue', label='Beta-sheets', linewidth=1, alpha=0.5)
            self.colGraph.canvas.ax.axvline(x=1629.5, color='blue', label='Beta-sheets', linewidth=11.5, alpha=0.5)
        self.colGraph.canvas.ax.invert_xaxis()
        self.colGraph.canvas.draw()

    def plot3D(self):
        self.colGraph.canvas.ax.clear()
        first_column = self.Columns[0]
        second_column = self.Columns[1]
        third_column = self.Columns[2]
        first_column -= 1
        second_column -= 1
        third_column -= 1
        x = self.t_matrix[:, first_column]
        y = self.t_matrix[:, second_column]
        z = self.t_matrix[:, third_column]
        for index in range(len(self.filenames)):
            if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                self.colGraph.canvas.ax.scatter(x[index], y[index], self.z[index], color="red")
            elif self.filenames[index][0] == 'N':
                self.colGraph.canvas.ax.scatter(x[index], y[index], z[index], color="blue")
            elif self.filenames[index][0] == 'D':
                self.colGraph.canvas.ax.scatter(x[index], y[index], self.z[index], color="green")
            elif (self.filenames[index][0] == 'O') or (self.filenames[index][0] == 'B'):
                self.colGraph.canvas.ax.scatter(x[index], y[index], self.z[index], color="black")
        self.colGraph.canvas.draw()


class MplCanvas(Canvas):
    def __init__(self, signal):
        dpi = 100
        self.fig = Figure(figsize=(1240/dpi, 600/dpi), dpi=dpi)
        if signal == 1 or signal == 2:
            self.ax = self.fig.add_subplot(111)
        elif signal == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            label.set_fontsize(16)
        super(MplCanvas, self).__init__(self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class MplWidget(QtWidgets.QWidget):
    def __init__(self, signal, parent=None):
        super().__init__()
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas(signal)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
