import os
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ColWin(QMainWindow):
    def __init__(self, parent, signal, filenames, t_pca, p_pca, t_der1, p_der1, t_der2, p_der2, tr_pca, pr_pca, waves,):
        super(ColWin, self).__init__(parent)
        print(os.path.abspath(os.curdir))
        loadUi("interface\\ColumnWindow.ui", self)
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
        Columns_temp = self.Columns_int.text().replace(' ', '').replace('.', ',').split(',')
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

    def prepare_data(self, *args):
        samples = {
            "D": [],
            "M": [],
            "N": [],
            "O": [],
        }
        for item in range(len(args)):
            for key in samples:
                samples[key].append([])

        for index in range(len(self.filenames)):
            for axis in range(len(args)):
                if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                    samples['M'][axis].append(args[axis][index])
                elif self.filenames[index][0] == 'N':
                    samples['N'][axis].append(args[axis][index])
                elif self.filenames[index][0] == 'D':
                    samples['D'][axis].append(args[axis][index])
                elif (self.filenames[index][0] == 'O') or (self.filenames[index][0] == 'B'):
                    samples['O'][axis].append(args[axis][index])
        return samples

    def plotScores(self):
        self.colGraph.canvas.ax.clear()
        x = self.t_matrix[:, self.Columns[0] - 1]
        y = self.t_matrix[:, self.Columns[1] - 1]

        samples = self.prepare_data(x, y, self.filenames)
        colors = {'M': 'red', 'N': 'blue', 'D': 'green', 'O': 'black'}
        markers = {'M': 'o', 'N': '^', 'D': '*', 'O': 's'}
        for key, value in samples.items():
            for index in range(len(self.filenames)):
                self.colGraph.canvas.ax.scatter(value[0], value[1],
                                                color=colors[key], marker=markers[key], alpha=1, zorder=10)
            for item in range(len(value[0])):
                self.colGraph.canvas.ax.annotate(value[-1][item], (value[0][item], value[1][item]))
        self.colGraph.canvas.draw()

    def plotLoadings(self):
        xs = self.waves_for_graph
        ys = list(self.p_matrix[:, self.Columns[0] - 1])

        self.colGraph.canvas.ax.clear()
        self.colGraph.canvas.ax.scatter(xs, ys, color="black", marker="o", s=12)

        wave = [1690, 1684, 1682.5, 1672, 1664, 1652.5, 1647, 1631, 1629.5]
        struct = ['Beta-turns', 'Beta-turns', 'Beta-sheets', 'Beta-turns', 'Beta-turns', 'Alpha-helices',
                  'Random-coil', 'Beta-sheets', 'Beta-sheets', ]
        color = ['green', 'green', 'blue', 'green', 'green', 'red', 'orange', 'blue', 'blue']
        width = [1, 1, 12.5, 1, 1, 3, 5, 2, 1, 11.5]
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

        if max(xs) >= 1652.5 and min(xs) <= 1629.5:
            for ind in range(len(wave)):
                self.colGraph.canvas.ax.axvline(x=wave[ind], color=color[ind], label=struct[ind],
                                                linewidth=width[ind], alpha=0.5)
            self.colGraph.canvas.ax.invert_xaxis()
        self.colGraph.canvas.draw()

    def plot3D(self):
        self.colGraph.canvas.ax.clear()
        x = self.t_matrix[:, self.Columns[0] - 1]
        y = self.t_matrix[:, self.Columns[1] - 1]
        z = self.t_matrix[:, self.Columns[2] - 1]
        samples = self.prepare_data(x, y, z, self.filenames)
        colors = {'M': 'red', 'N': 'blue', 'D': 'green', 'O': 'black'}
        markers = {'M': 'o', 'N': '^', 'D': '*', 'O': 's'}
        for key, value in samples.items():
            self.colGraph.canvas.ax.scatter(samples[key][0], samples[key][1], samples[key][2],
                                            color=colors[key], marker=markers[key], alpha=1)
            for item in range(len(value[0])):
                self.colGraph.canvas.ax.annotate(value[-1][item], (value[0][item], value[1][item]))
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
