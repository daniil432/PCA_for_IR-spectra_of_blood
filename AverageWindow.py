import copy
import math
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


class AverWin(QDialog):
    def __init__(self, ratio_waves):
        super(AverWin, self).__init__()
        loadUi("C:\\PCA_with_R\\interface\\AverageWindow.ui", self)
        self.tab_mm.RatioWidget = RatioWidgetAverage(ratio_waves, 1, self.RatioWidget)
        self.tab_non_mm.RatioWidget = RatioWidgetAverage(ratio_waves, 2, self.RatioWidget_2)
        self.tab_wave.WaveWidget = WaveWidgetAverage(ratio_waves, self.WaveWidget)
        self.CloseButton.clicked.connect(self.close)


class MplCanvasAverage(Canvas):
    def __init__(self, type_of_graph, data_len):
        dpi = 100
        if type_of_graph == 'polar':
            self.fig = Figure(figsize=(1640/dpi, 750/dpi), dpi=dpi)
            self.ax = self.fig.add_subplot(111, projection='polar')
        elif type_of_graph == 'errorbar':
            self.fig, self.ax = plt.subplots(1, data_len, figsize=(1640/dpi, 750/dpi), constrained_layout=True)
            for i in range(len(self.ax)):
                self.ax[i].xaxis.set_visible(False)
                self.ax[i].yaxis.set_visible(True)
                self.ax[i].tick_params(labelsize=16, direction='in')
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
        self.toolbar = NavigationToolbar(canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(canvas)
        self.vbl.addWidget(canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        labels = ['A1/Min1', 'A1/s.ch.', 'A1/Min2', 'A1/Tyr', 'A1/Min3', "A1/A2'",
                  'Min1/s.ch.', 'Min1/Min2', 'Min1/Tyr', 'Min1/Min3', "Min1/A2'",
                  's.ch./Min2', 's.ch./Tyr', 's.ch./Min3', "s.ch./A2'",
                  'Min2/Tyr', 'Min2/Min3', "Min2/A2'",
                  'Tyr/Min3', "Tyr/A2'",
                  "Min3/A2'"]
        #drop_ = [20, 19, 18, 17, 16, 15, 9, 6, 5, 4, 3, 2] # Временно
        #for ind in range(len(drop_)): # Временно
        #    labels.pop(drop_[ind]) # Временно
        #    self.result_d = np.delete(self.result_d, drop_[ind]) # Временно
        #    self.result_p = np.delete(self.result_p, drop_[ind]) # Временно
        #    self.result_n = np.delete(self.result_n, drop_[ind]) # Временно
        #    error_radial_d = np.delete(error_radial_d, drop_[ind]) # Временно
        #    error_radial_p = np.delete(error_radial_p, drop_[ind]) # Временно
        #    error_radial_n = np.delete(error_radial_n, drop_[ind]) # Временно
        theta = np.linspace(start=0, stop=2 * np.pi, num=len(result_d) - 1, endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))
        canvas.ax.plot(theta, result_d, linewidth=2, linestyle='-', color="green")
        canvas.ax.bar(theta, result_d, linewidth=0, yerr=error_radial_d, capsize=0.0001, color="green",
                           fill=None, ecolor="green", alpha=0.8)
        if mode == 1:
            if None in result_p:
                pass
            else:
                canvas.ax.plot(theta, result_p, linewidth=2, linestyle='--', color="red")
                canvas.ax.bar(theta, result_p, linewidth=0, yerr=error_radial_p, capsize=0.0001, color="red",
                              fill=None, ecolor="red", alpha=0.8)
        else:
            if None in result_n:
                pass
            else:
                canvas.ax.plot(theta, result_n, linewidth=2, linestyle='--', color="blue")
                canvas.ax.bar(theta, result_n, linewidth=0, yerr=error_radial_n, capsize=0.0001, color="blue",
                              fill=None, ecolor="blue", alpha=0.8)
        _ran = [*range(0, 360, math.floor(360 / len(labels)))]
        if len(_ran) > len(labels):
            _ran.pop(-1)
        canvas.ax.set_thetagrids(_ran, labels, fontsize=16)
        plt.yticks(np.arange(0, 1.5, 0.2), fontsize=16)
        legend_label = {1: "Secretory MM patients", 2: "Non-secretory MM patients"}
        canvas.ax.legend([Line2D([0], [0], linestyle='-', color='g', lw=3),
                               Line2D([0], [0], linestyle='--', color=['r' if mode == 1 else 'b'][0], lw=3)],
                              ['Healthy donors', legend_label[mode]], prop={'size': 16},
                              loc='upper center', bbox_to_anchor=(1.4, 0.05), fancybox=True, shadow=True)
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

        cat_par = ['Amide-I', 'Min 1', 'Side chains', 'Min2', 'Tyr', 'Min 3', "Amide-II'"]
        #drop_ = [6]
        #for ind in range(len(drop_)):
        #    cat_par.pop(drop_[ind])
        #    g1 = np.delete(g1, drop_[ind]) # Временно
        #    g2 = np.delete(g2, drop_[ind]) # Временно
        #    g3 = np.delete(g3, drop_[ind]) # Временно
        canvas = MplCanvasAverage('errorbar', len(g1))
        self.toolbar = NavigationToolbar(canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
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
                                      bottom=bottom[index], yerr=error_d[index], ecolor="black", alpha=0.6,
                                      color='g', edgecolor="black", linewidth=0.1, capsize=6)
            canvas.ax[index].bar(1, g2[index] - bottom[index], width=0.3, bottom=bottom[index],
                                      yerr=error_p[index], ecolor="black", alpha=0.6, color='r',
                                      edgecolor="black", linewidth=0.1, capsize=6, hatch='.')
            if (g3[index] == 0) or (g3[index] == None) or (g3[index] == []):
                pass
            else:
                canvas.ax[index].bar(1 + width, g3[index] - bottom[index], width=0.3,
                                          bottom=bottom[index], yerr=error_n[index], ecolor="black", alpha=0.6,
                                          color='b', edgecolor="black", linewidth=0.1, capsize=6, hatch='/')
            canvas.ax[index].set_title(fontsize=16, label=cat_par[index])
