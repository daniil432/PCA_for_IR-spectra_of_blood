import copy
import math
import random
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from math import acos, sqrt


class AverWin(QMainWindow):
    def __init__(self, parent, ratio_waves):
        super(AverWin, self).__init__(parent)
        loadUi("C:\\PCA_with_R\\interface\\AverageWindow.ui", self)
        self.parent = parent
        self.tab_mm.RatioWidget = RatioWidgetAverage(ratio_waves, 1, self.RatioWidget)
        self.tab_non_mm.RatioWidget = RatioWidgetAverage(ratio_waves, 2, self.RatioWidget_2)
        self.tab_wave.WaveWidget = WaveWidgetAverage(ratio_waves, self.WaveWidget)
        self.CloseButton.clicked.connect(self.closeEvent)
        QAction("Quit", self).triggered.connect(self.closeEvent)


def closeEvent(self, event):
        self.parent.show()
        self.close()


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
    def __init__(self, ratio_waves, mode, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        error_radial_d = np.append(ratio_waves['D'][2], ratio_waves['D'][2][0])
        if 'M' in ratio_waves.keys():
            if ratio_waves['M'][2] is not None:
                error_radial_p = np.append(ratio_waves['M'][2], ratio_waves['M'][2][0])
            else:
                error_radial_p = [None] * len(error_radial_d)
        else:
            error_radial_p = [None] * len(error_radial_d)
        if 'N' in ratio_waves.keys():
            if ratio_waves['N'][2] is not None:
                error_radial_n = np.append(ratio_waves['N'][2], ratio_waves['N'][2][0])
            else:
                error_radial_n = [None] * len(error_radial_d)
        else:
            error_radial_n = [None] * len(error_radial_d)

        result_d = np.append(ratio_waves['D'][0], ratio_waves['D'][0][0])
        if 'M' in ratio_waves.keys():
            if ratio_waves['M'][0] is not None:
                result_p = np.append(ratio_waves['M'][0], ratio_waves['M'][0][0])
            else:
                result_p = [None] * len(result_d)
        else:
            result_p = [None] * len(result_d)
        if 'N' in ratio_waves.keys():
            if ratio_waves['N'][0] is not None:
                result_n = np.append(ratio_waves['N'][0], ratio_waves['N'][0][0])
            else:
                result_n = [None] * len(result_d)
        else:
            result_n = [None] * len(result_d)

        canvas = MplCanvasAverage('polar', len(result_d)-1)
        labels = ['M$_{I}$/N$_{1}$', 'M$_{I}$/M$_{S}$', 'M$_{I}$/N$_{2}$', 'M$_{I}$/M$_{T}$', 'M$_{I}$/N$_{3}$', "M$_{I}$/M$_{II}$",
                  'M$_{S}$/N$_{1}$', 'N$_{1}$/N$_{2}$', 'N$_{1}$/M$_{T}$', 'N$_{1}$/N$_{3}$', "M$_{II}$/N$_{1}$",
                  'M$_{S}$/N$_{2}$', 'M$_{S}$/M$_{T}$', 'M$_{S}$/N$_{3}$', "M$_{II}$/M$_{S}$",
                  'M$_{T}$/N$_{2}$', 'N$_{2}$/N$_{3}$', "M$_{II}$/N$_{2}$",
                  'M$_{T}$/N$_{3}$', "M$_{II}$/M$_{T}$",
                  "M$_{II}$/N$_{3}$"]

        # drop_ = [20, 19, 18, 17, 16, 15, 9, 6, 5, 4, 3, 2] # Временно убираем элементы из обзора для статьи...
        # for ind in range(len(drop_)): # Временно
        #     labels.pop(drop_[ind]) # Временно
        #     result_d = np.delete(result_d, drop_[ind]) # Временно
        #     result_p = np.delete(result_p, drop_[ind]) # Временно
        #     result_n = np.delete(result_n, drop_[ind]) # Временно
        #     error_radial_d = np.delete(error_radial_d, drop_[ind]) # Временно
        #     error_radial_p = np.delete(error_radial_p, drop_[ind]) # Временно
        #     error_radial_n = np.delete(error_radial_n, drop_[ind]) # Временно
        theta = np.linspace(start=0, stop=2 * np.pi, num=len(result_d) - 1, endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))
        canvas.ax.plot(theta, result_d, linewidth=2, linestyle='-', color="green")
        canvas.ax.bar(theta, result_d, linewidth=0, yerr=error_radial_d, capsize=0.00008, color="green",
                      fill=None, ecolor="green", alpha=0.8)
        if mode == 1:
            if None in result_p:
                pass
            else:
                canvas.ax.plot(theta, result_p, linewidth=2, linestyle='--', color="red")
                canvas.ax.bar(theta, result_p, linewidth=0, yerr=error_radial_p, capsize=0.00008, color="red",
                              fill=None, ecolor="red", alpha=0.8)
        else:
            if None in result_n:
                pass
            else:
                canvas.ax.plot(theta, result_n, linewidth=2, linestyle='--', color="blue")
                canvas.ax.bar(theta, result_n, linewidth=0, yerr=error_radial_n, capsize=0.00008, color="blue",
                              fill=None, ecolor="blue", alpha=0.8)
        _ran = [*range(0, 360, math.floor(360 / len(labels)))]
        if len(_ran) > len(labels):
            _ran.pop(-1)
        canvas.ax.set_thetagrids(_ran, labels, fontsize=10)
        plt.yticks(np.arange(0, 1.5, 0.2), fontsize=10)
        legend_label = {1: "Пациенты с секр. ММ", 2: "Пациенты с не секр. ММ"}
        canvas.ax.legend([Line2D([0], [0], linestyle='-', color='g', lw=3),
                          Line2D([0], [0], linestyle='--', color=['r' if mode == 1 else 'b'][0], lw=3)],
                         ['Здоровые доноры', legend_label[mode]], prop={'size': 12},
                         loc='upper center', bbox_to_anchor=(0.005, 1.165), fancybox=True, shadow=True)
        canvas.ax.set(facecolor='#f3f3f3')
        canvas.ax.set_theta_offset(np.pi / 2)
        canvas.ax.set_theta_direction(-1)
        pl = canvas.ax.yaxis.get_gridlines()
        for line in pl:
            line.get_path()._interpolation_steps = 5

        def correct_errorbar(ax, barlen=0.05, errorline=1, color='black', mode=1):
            x, y = ax.lines[errorline].get_data()
            del ax.lines[errorline]
            for i in range(len(y)):
                r = sqrt(barlen * barlen / 4 + y[i] * y[i])
                dt = acos((y[i]) / (r))
                newline = Line2D([x[i] - dt, x[i] + dt], [r, r], color=color, linewidth=2, zorder=50+errorline, alpha=0.8)
                ax.add_line(newline)

        _indexes = [5, 4, 2, 1]
        _color = {1: ['red', 'red', 'green', 'green'], 2: ['blue', 'blue', 'green', 'green']}
        try:
            for i in range(len(_indexes)):
                correct_errorbar(canvas.ax, barlen=0.1, errorline=_indexes[i], color=_color[mode][i], mode=mode)
        except:
            pass
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(canvas)
        self.toolbar = NavigationToolbar(canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        # r = random.randint(1, 30)
        # canvas.fig.savefig(f'fig_en_{r}.tiff')
        # canvas.fig.savefig(f'fig_en_{r}.eps')


class WaveWidgetAverage(QtWidgets.QWidget):
    def __init__(self, ratio_waves, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        g1 = ratio_waves['D'][1]
        if 'M' in ratio_waves.keys():
            g2 = ratio_waves['M'][1]
        else:
            g2 = [None] * len(g1)
        if 'N' in ratio_waves.keys():
            g3 = ratio_waves['N'][1]
        else:
            g3 = [None] * len(g1)

        error_d = np.array(ratio_waves['D'][3]).T
        if 'M' in ratio_waves.keys():
            error_p = np.array(ratio_waves['M'][3]).T
        else:
            error_p = [None] * len(g1)
        if 'N' in ratio_waves.keys():
            error_n = np.array(ratio_waves['N'][3]).T
        else:
            error_n = [None] * len(g1)

        cat_par = ['M$_{I}$', 'N$_{1}$', 'M$_{S}$', 'N$_{2}$', 'M$_{T}$', 'N$_{3}$', "M$_{II}$"]
        # drop_ = [6] # Временно убираем элементы из обзора для статьи
        # for ind in range(len(drop_)): # Временно
        #     cat_par.pop(drop_[ind]) # Временно
        #     g1 = np.delete(g1, drop_[ind]) # Временно
        #     g2 = np.delete(g2, drop_[ind]) # Временно
        #     g3 = np.delete(g3, drop_[ind]) # Временно
        canvas = MplCanvasAverage('errorbar', len(g1))
        width = 0.3
        bottom = []
        for wave in range(len(g1)):
            if None not in g2 and None not in g3:
                min_wave = copy.copy(min(g1[wave], g2[wave], g3[wave]))
            elif None not in g2 and None in g3:
                min_wave = copy.copy(min(g1[wave], g2[wave]))
            elif None not in g3 and None in g2:
                min_wave = copy.copy(min(g1[wave], g3[wave]))
            bottom.append(min_wave - 1.5)
        for index in range(len(g1)):
            canvas.ax[index].bar(1 - width, g1[index] - bottom[index], width=0.3,
                                 bottom=bottom[index], yerr=error_d[index], ecolor="black", alpha=0.6, color='g',
                                 edgecolor="black", linewidth=0.1, capsize=4, error_kw={'elinewidth': 1})
            canvas.ax[index].bar(1, g2[index] - bottom[index], width=0.3, bottom=bottom[index], yerr=error_p[index],
                                 ecolor="black", alpha=0.6, color='r', edgecolor="black", linewidth=0.1, capsize=4,
                                 hatch='.', error_kw={'elinewidth': 1})
            if (g3[index] != 0) or (g3[index] is not None) or (g3[index] != []):
                canvas.ax[index].bar(1 + width, g3[index] - bottom[index], width=0.3,
                                     bottom=bottom[index], yerr=error_n[index], ecolor="black", alpha=0.6, color='b',
                                     edgecolor="black", linewidth=0.1, capsize=4, hatch='/', error_kw={'elinewidth': 1})
            canvas.ax[index].set_title(label=cat_par[index], fontsize=9)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(canvas)
        self.toolbar = NavigationToolbar(canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        plt.tight_layout()
        plt.legend(labels=['Зд. доноры', "Пациенты с секр. ММ", "Пациенты с не секр. ММ"], loc='lower center',
                   ncol=3, bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.03, -0.00, 1, 2), framealpha=1)
        # r = random.randint(1, 30)
        # canvas.fig.savefig(f'fig_en_{r}.tiff')
        # canvas.fig.savefig(f'fig_en_{r}.eps')
