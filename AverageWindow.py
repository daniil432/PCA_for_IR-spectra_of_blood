import copy
import math
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from math import acos, sqrt
from PyQt5 import QtCore, QtGui, QtWidgets


class AverWin(QMainWindow):
    def __init__(self, parent, ratio_waves):
        super(AverWin, self).__init__(parent)
        loadUi("interface\\AverageWindow.ui", self)
        _translate = QtCore.QCoreApplication.translate
        self.parent = parent
        for key in list(ratio_waves.keys()):
            if key != 'D':
                setattr(self, f"tab_{key}", self.newTab(key))
                self.tabWidget.addTab(getattr(self, f"tab_{key}"), f"tab_{key}")
                self.label_2.setText(_translate("MainWindow",
                                                "<html><head/><body><p align=\"center\">"
                                                "Средние значения отношений поглощения для здоровых доноров и "
                                                "иной рассматриваемой группы образцов. Зелёным цветом "
                                                "обозначены доноры, другим цветом - выбранная группа."
                                                "</p></body></html>"))
                self.tabWidget.setTabText(self.tabWidget.indexOf(getattr(self, f"tab_{key}")),
                                          _translate("MainWindow", f"Ratio_{key}"))
                getattr(self, f"tab_{key}").RatioWidget = RatioWidgetAverage(
                    {'D': copy.deepcopy(ratio_waves['D']), key: copy.deepcopy(ratio_waves[key])}, key, self.RatioWidget)
        self.tab_wave.WaveWidget = WaveWidgetAverage(ratio_waves, self.WaveWidget)
        self.CloseButton.clicked.connect(self.closeEvent)
        QAction("Quit", self).triggered.connect(self.closeEvent)

    def closeEvent(self, event):
        self.parent.show()
        self.close()

    def newTab(self, name):
        setattr(self, f"tab_{name}", QtWidgets.QWidget())
        getattr(self, f"tab_{name}").setObjectName(f"tab_{name}")
        self.label_3 = QtWidgets.QLabel(getattr(self, f"tab_{name}"))
        self.label_3.setGeometry(QtCore.QRect(160, 10, 1400, 90))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.RatioWidget = QtWidgets.QWidget(getattr(self, f"tab_{name}"))
        self.RatioWidget.setGeometry(QtCore.QRect(40, 100, 1661, 771))
        self.RatioWidget.setObjectName("RatioWidget")
        self.label_2 = QtWidgets.QLabel(getattr(self, f"tab_{name}"))
        self.label_2.setGeometry(QtCore.QRect(150, 0, 1400, 90))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        return getattr(self, f"tab_{name}")



class MplCanvasAverage(Canvas):
    def __init__(self, type_of_graph, data_len):
        dpi = 100
        if type_of_graph == 'polar':
            self.fig = Figure(figsize=(1400/dpi, 700/dpi), dpi=dpi) # for saving dpi = 600 and no figsize
            self.ax = self.fig.add_subplot(111, projection='polar')
            self.ax.tick_params(direction='out', zorder=1)
        elif type_of_graph == 'errorbar':
            self.fig, self.ax = plt.subplots(1, data_len, dpi=dpi, figsize=(1400/dpi, 700/dpi))
            for i in range(len(self.ax)):
                self.ax[i].xaxis.set_visible(False)
                self.ax[i].yaxis.set_visible(True)
                self.ax[i].tick_params(labelsize=7, direction='in')
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class RatioWidgetAverage(QtWidgets.QWidget):
    def __init__(self, ratio_waves, key_graph, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        # Длина первого подсписка под первым ключом словаре
        list_of_keys = list(ratio_waves.keys())
        canv_size = len(ratio_waves[list_of_keys[0]][0])
        canvas = MplCanvasAverage('polar', canv_size)
        labels = ['M$_{I}$/N$_{1}$', 'M$_{I}$/M$_{S}$', 'M$_{I}$/N$_{2}$', 'M$_{I}$/M$_{T}$', 'M$_{I}$/N$_{3}$',
                  "M$_{I}$/M$_{II}$",
                  'M$_{S}$/N$_{1}$', 'N$_{1}$/N$_{2}$', 'N$_{1}$/M$_{T}$', 'N$_{1}$/N$_{3}$', "M$_{II}$/N$_{1}$",
                  'M$_{S}$/N$_{2}$', 'M$_{S}$/M$_{T}$', 'M$_{S}$/N$_{3}$', "M$_{II}$/M$_{S}$",
                  'M$_{T}$/N$_{2}$', 'N$_{2}$/N$_{3}$', "M$_{II}$/N$_{2}$",
                  'M$_{T}$/N$_{3}$', "M$_{II}$/M$_{T}$",
                  "M$_{II}$/N$_{3}$"]

        theta = np.linspace(start=0, stop=2 * np.pi, num=canv_size, endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))

        linestyles = {'D': '-', 'M': '--', 'N': '-.', 'O': ':', 'B': 'solid', 'U': 'dashed', }
        colors = {'D': "green", 'M': "red", 'N': "blue", 'O': "black", 'B': "orange", 'U': "purple", }
        ledend_labels = {'D': "Здоровые доноры", 'M': "Пациенты с секр. ММ", 'N': "Пациенты с не секр. ММ",
                         'O': "Тестовые доноры", 'B': "Тестовые доноры", 'U': "Неизвестные образцы", }

        for key in ratio_waves.keys():
            ratio_waves[key][0].append(ratio_waves[key][0][0])
            ratio_waves[key][2].append(ratio_waves[key][2][0])
            canvas.ax.plot(theta, ratio_waves[key][0], linewidth=2, linestyle=linestyles[key], color=colors[key])
            canvas.ax.bar(theta, ratio_waves[key][0], linewidth=0, yerr=ratio_waves[key][2], capsize=0.00008,
                          color=colors[key], fill=None, ecolor=colors[key], alpha=0.8)

        _ran = [*range(0, 360, math.floor(360 / len(labels)))]
        if len(_ran) > len(labels):
            _ran.pop(-1)
        canvas.ax.set_thetagrids(_ran, labels, fontsize=10)
        plt.yticks(np.arange(0, 1.5, 0.2), fontsize=10)

        canvas.ax.legend([Line2D([0], [0], linestyle=linestyles['D'], color=colors['D'], lw=3),
                          Line2D([0], [0], linestyle=linestyles[key_graph], color=colors[key_graph], lw=3)],
                         [ledend_labels['D'], ledend_labels[key_graph]], prop={'size': 12},
                         loc='upper center', bbox_to_anchor=(0.005, 1.165), fancybox=True, shadow=True)

        canvas.ax.set(facecolor='#f3f3f3')
        canvas.ax.set_theta_offset(np.pi / 2)
        canvas.ax.set_theta_direction(-1)
        pl = canvas.ax.yaxis.get_gridlines()
        for line in pl:
            line.get_path()._interpolation_steps = 5

        def correct_errorbar(ax, barlen=0.05, errorline=1, color='black'):
            x, y = ax.lines[errorline].get_data()
            del ax.lines[errorline]
            for i in range(len(y)):
                r = sqrt(barlen * barlen / 4 + y[i] * y[i])
                dt = acos((y[i]) / (r))
                newline = Line2D([x[i] - dt, x[i] + dt], [r, r], color=color, linewidth=2, zorder=50+errorline, alpha=0.8)
                ax.add_line(newline)

        _indexes = [5, 4, 2, 1]
        _colors = [colors[key_graph], colors[key_graph], colors['D'], colors['D'], ]
        try:
            for i in range(len(_indexes)):
                correct_errorbar(canvas.ax, barlen=0.1, errorline=_indexes[i], color=_colors[i])
        except:
            pass
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(canvas)
        self.toolbar = NavigationToolbar(canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class WaveWidgetAverage(QtWidgets.QWidget):
    def __init__(self, ratio_waves, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        key_len = len(ratio_waves['D'][1])
        canvas = MplCanvasAverage('errorbar', key_len)
        pos = [1/key_len + 1/key_len * 2*k for k in range(0, key_len)]
        cat_par = ['M$_{I}$', 'N$_{1}$', 'M$_{S}$', 'N$_{2}$', 'M$_{T}$', 'N$_{3}$', "M$_{II}$"]
        hatches = {'D': None, 'M': '.', 'N': '/', 'O': '\\', 'B': 'o', 'U': '*', }
        colors = {'D': "green", 'M': "red", 'N': "blue", 'O': "black", 'B': "orange", 'U': "purple", }
        ledend_labels = {'D': "Здоровые доноры", 'M': "Пациенты с секр. ММ", 'N': "Пациенты с не секр. ММ",
                         'O': "Тестовые доноры", 'B': "Тестовые доноры", 'U': "Неизвестные образцы", }

        bottom = []
        for index in range(key_len):
            bottom.append(min([ratio_waves[k][1][index] for k in list(ratio_waves.keys())])-1.5)

        for index in range(len(ratio_waves['D'][1])):
            for key, value in ratio_waves.items():
                canvas.ax[index].bar(pos[list(ratio_waves.keys()).index(key)], value[1][index] - bottom[index],
                                     bottom=bottom[index], width=0.3, yerr=value[3][index], ecolor="black", alpha=0.6,
                                     color=colors[key], capsize=4, hatch=hatches[key], edgecolor="black", linewidth=0.1,
                                     error_kw={'elinewidth': 1})

        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(canvas)
        self.toolbar = NavigationToolbar(canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        plt.tight_layout()
        plt.legend(labels=[i for i in [ledend_labels[k] for k in list(ratio_waves.keys())]], loc='lower center',
                   ncol=key_len, bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.03, -0.00, 1, 2), framealpha=1)
