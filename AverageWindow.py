from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from numpy import array
import matplotlib.pyplot as plt
import numpy as np


class Ui_AverageWindow(object):
    def Signal(self, main):
        self.main = main


    def Home(self):
        self.AverageWindow.close()
        self.SecondWindow.show()


    def setupUi(self, AverageWindow, SecondWindow):
        self.AverageWindow = AverageWindow
        self.SecondWindow = SecondWindow
        AverageWindow.setObjectName("AverageWindow")
        AverageWindow.resize(973, 588)
        self.centralwidget = QtWidgets.QWidget(AverageWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.RatioWindow = RatioWidget(self.main, self.centralwidget)
        self.RatioWindow.setGeometry(QtCore.QRect(20, 110, 461, 391))
        self.RatioWindow.setObjectName("RatioWindow")
        self.WaveWindow = WaveWidget(self.main, self.centralwidget)
        self.WaveWindow.setGeometry(QtCore.QRect(490, 110, 461, 391))
        self.WaveWindow.setObjectName("WaveWindow")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(26, 12, 921, 91))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.CloseButton = QtWidgets.QPushButton(self.centralwidget)
        self.CloseButton.setGeometry(QtCore.QRect(450, 530, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.CloseButton.setFont(font)
        self.CloseButton.setObjectName("CloseButton")
        AverageWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(AverageWindow)
        self.statusbar.setObjectName("statusbar")
        AverageWindow.setStatusBar(self.statusbar)

        self.retranslateUi(AverageWindow)
        QtCore.QMetaObject.connectSlotsByName(AverageWindow)


    def retranslateUi(self, AverageWindow):
        _translate = QtCore.QCoreApplication.translate
        AverageWindow.setWindowTitle(_translate("AverageWindow", "MainWindow"))
        self.label.setText(_translate("AverageWindow", "<html><head/><body><p align=\"center\"><span style=\" "
                                                       "font-size:14pt;\">"
                                                       "Средние значения отношений поглощения и длин волн для здоровых"
                                                       " доноров и пациентов с ММ. Зелёным цветом обозначены доноры, "
                                                       "красным цветом - больные с секретирующей ММ, синим цветом - "
                                                       "больные с несекретирующей ММ.</span></p></body></html>"))
        self.CloseButton.setText(_translate("AverageWindow", "ok"))


class MplCanvas(Canvas):
    def __init__(self, type_of_graph):
        if type_of_graph == 'polar':
            self.fig = Figure(figsize=(14, 14), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='polar')
        elif type_of_graph == 'errorbar':
            self.fig, self.ax = plt.subplots(1, 5, figsize=(14, 14), constrained_layout=True)
            for i in range(len(self.ax)):
                self.ax[i].xaxis.set_visible(False)
                self.ax[i].yaxis.set_visible(True)
                self.ax[i].tick_params(labelsize=8, direction='in')
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class RatioWidget(QtWidgets.QWidget):
    def __init__(self, main, parent=None):
        self.main = main
        error_radial = [0.5, 0.4, 0.001, 0.07, 0.01, 0.001, 0.001, 0.2, 0.03, 0.001]
        for i in range(len(error_radial)):
            error_radial[i] = error_radial[i] * self.main.normal[i]
        self.result_d = np.append(self.main.result_d, self.main.result_d[0])
        self.result_p = np.append(self.main.result_p, self.main.result_p[0])
        self.result_n = np.append(self.main.result_n, self.main.result_n[0])
        self.error_radial = np.append(error_radial, error_radial[0])
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas('polar')
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


class WaveWidget(QtWidgets.QWidget):
    def __init__(self, main, parent=None):
        self.main = main
        g1 = self.main.result_waves_d
        g2 = self.main.result_waves_p
        g3 = self.main.result_waves_n
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas('errorbar')
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        cat_par = ['Amide-I', 'Min 1-2', 'Amide-II', 'Min 2-3', 'Amide-III']
        width = 0.3
        error_d = array([0.89, 0.364, 0.625, 0.483, 0.246]).T
        error_p = array([0.1, 0.3, 0.2, 0.4, 0.5]).T
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
