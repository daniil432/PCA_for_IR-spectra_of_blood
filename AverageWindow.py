from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class AverWin(QDialog):
    def __init__(self, SpeAn):
        self.SpeAn = SpeAn
        super(AverWin, self).__init__()
        loadUi("C:\\PCA_with_R\\AverageWindow.ui", self)
        self.RatioWidget = RatioWidgetAverage(self.SpeAn, self.RatioWidget)
        self.WaveWidget = WaveWidgetAverage(self.SpeAn, self.WaveWidget)
        self.CloseButton.clicked.connect(self.close)


class MplCanvasAverage(Canvas):
    def __init__(self, type_of_graph):
        if type_of_graph == 'polar':
            dpi = 100
            self.fig = Figure(figsize=(780/dpi, 720/dpi), dpi=dpi)
            self.ax = self.fig.add_subplot(111, projection='polar')
        elif type_of_graph == 'errorbar':
            self.fig, self.ax = plt.subplots(1, 5, figsize=(780/100, 720/100), constrained_layout=True)
            for i in range(len(self.ax)):
                self.ax[i].xaxis.set_visible(False)
                self.ax[i].yaxis.set_visible(True)
                self.ax[i].tick_params(labelsize=8, direction='in')
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class RatioWidgetAverage(QtWidgets.QWidget):
    def __init__(self, SpeAn, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.SpeAn = SpeAn
        error_radial = [0.5, 0.4, 0.001, 0.07, 0.01, 0.001, 0.001, 0.2, 0.03, 0.001]
        for i in range(len(error_radial)):
            error_radial[i] = error_radial[i] * self.SpeAn.normal[i]
        self.result_d = np.append(self.SpeAn.result_d, self.SpeAn.result_d[0])
        self.result_p = np.append(self.SpeAn.result_p, self.SpeAn.result_p[0])
        self.result_n = np.append(self.SpeAn.result_n, self.SpeAn.result_n[0])
        self.error_radial = np.append(error_radial, error_radial[0])
        self.canvas = MplCanvasAverage('polar')
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        labels = ['A1/M1', 'A1/A2', 'A1/M2', 'A1/A3', 'M1/A2', 'M1/M2', 'M1/A3', 'A2/M2', 'A2/A3', 'M2/A3']
        self.theta = np.linspace(start=0, stop=2 * np.pi, num=len(self.result_d) - 1, endpoint=False)
        self.theta = np.concatenate((self.theta, [self.theta[0]]))
        self.canvas.ax.errorbar(self.theta, self.result_d, linewidth=2, xerr=0, yerr=self.error_radial, color="green",
                                ecolor='black')
        self.canvas.ax.errorbar(self.theta, self.result_p, linewidth=2, xerr=0, yerr=0, color="red")
        self.canvas.ax.errorbar(self.theta, self.result_n, linewidth=2, xerr=0, yerr=0, color="blue")
        self.canvas.ax.set_thetagrids(range(0, 360, int(360 / len(labels))), labels)
        plt.yticks(np.arange(0, 1.5, 0.2), fontsize=8)
        self.canvas.ax.set(facecolor='#f3f3f3')
        self.canvas.ax.set_theta_offset(np.pi / 2)

        pl = self.canvas.ax.yaxis.get_gridlines()
        for line in pl:
            line.get_path()._interpolation_steps = 5


class WaveWidgetAverage(QtWidgets.QWidget):
    def __init__(self, SpeAn, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.SpeAn = SpeAn
        g1 = self.SpeAn.result_waves_d
        g2 = self.SpeAn.result_waves_p
        g3 = self.SpeAn.result_waves_n
        self.canvas = MplCanvasAverage('errorbar')
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        cat_par = ['Amide-I', 'Min 1-2', 'Amide-II', 'Min 2-3', 'Amide-III']
        width = 0.3
        error_d = np.array([0.89, 0.364, 0.625, 0.483, 0.246]).T
        error_p = np.array([0.1, 0.3, 0.2, 0.4, 0.5]).T
        bottom = [1638.5, 1595.5, 1569.5, 1503.5, 1448.5]
        for index in range(len(g1)):
            self.canvas.ax[index].bar(1 - width, g1[index] - bottom[index], width=0.3,
                                      bottom=bottom[index],
                                      yerr=error_d[index], ecolor="black", alpha=0.6, color='g',
                                      edgecolor="blue",
                                      linewidth=0.1)
            self.canvas.ax[index].bar(1, g2[index] - bottom[index], width=0.3, bottom=bottom[index],
                                      yerr=error_p[index],
                                      ecolor="black", alpha=0.6, color='r', edgecolor="blue",
                                      linewidth=0.1)
            if (g3[index] == 0) or (g3[index] == None) or (g3[index] == []):
                pass
            else:
                self.canvas.ax[index].bar(1 + width, g3[index] - bottom[index], width=0.3,
                                          bottom=bottom[index],
                                          yerr=error_p[index], ecolor="black", alpha=0.6, color='b',
                                          edgecolor="blue",
                                          linewidth=0.1)
            self.canvas.ax[index].set_title(fontsize=8, label=cat_par[index])