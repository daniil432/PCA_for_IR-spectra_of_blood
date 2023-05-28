import copy
import math
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from math import acos, sqrt


class PatWin(QDialog):
    def __init__(self, parent, ratio, waves, ratio_waves, filenames):
        super(PatWin, self).__init__(parent)
        loadUi("interface\\PatientWindow.ui", self)
        self.parent = parent
        self.ratio = ratio
        self.waves = waves
        self.filenames = filenames
        self.averD = ratio_waves['D'][0]
        self.waveD = ratio_waves['D'][1]
        self.errors = ratio_waves['D'][2]
        self.w_errors = ratio_waves['D'][3]
        self.errors_M = ratio_waves['M'][2]
        self.errors_N = ratio_waves['N'][2]
        self.w_errors_M = ratio_waves['M'][3]
        self.w_errors_N = ratio_waves['N'][3]
        self.RatioWidget = RatioWidgetPatient(len(self.waveD), self.RatioWidget)
        self.WaveWidget = WaveWidgetPatient(len(self.waveD), self.WaveWidget)
        self.AcceptButton.clicked.connect(self.showGraph)
        self.Patient_int.returnPressed.connect(self.AcceptButton.click)
        self.CloseButton.clicked.connect(self.closeEvent)
        QAction("Quit", self).triggered.connect(self.closeEvent)

    def closeEvent(self, event):
        self.parent.show()
        self.close()

    def showGraph(self):
        self.RatioWidget.canvas.ax.clear()
        self.WaveWidget.canvas.axs[0].clear()
        self.WaveWidget.canvas.axs[1].clear()
        self.WaveWidget.canvas.axs[2].clear()
        self.WaveWidget.canvas.axs[3].clear()
        self.WaveWidget.canvas.axs[4].clear()
        self.WaveWidget.canvas.axs[5].clear()
        self.WaveWidget.canvas.axs[6].clear()

        if self.Patient_int.text() != '':
            Patients_temp = self.Patient_int.text()
            Patients_temp = Patients_temp.replace(' ', '')
            Patients_temp = Patients_temp.replace('.', ',')
            input_temp = Patients_temp.split(',')
            self.Patients = []
            for j in input_temp:
                self.Patients.append(int(j))
            if self.Patients == []:
                pass
        else:
            self.Patients = [1, 2]

        secr_intensities = []
        nesecr_intensities = []
        pat_wave = []
        non_wave = []
        secr_filenames = []
        nesecr_filenames = []
        for name in range(len(self.filenames)):
            if self.filenames[name][0] == 'M' or self.filenames[name][0] == 'S':
                secr_intensities.append(self.ratio[name][:21])
                secr_filenames.append(self.filenames[name])
                pat_wave.append(self.waves[name])
            elif self.filenames[name][0] == 'N':
                nesecr_intensities.append(self.ratio[name][:21])
                non_wave.append(self.waves[name])
                nesecr_filenames.append(self.filenames[name])
        labels = ['M$_{I}$/N$_{1}$', 'M$_{I}$/M$_{S}$', 'M$_{I}$/N$_{2}$', 'M$_{I}$/M$_{T}$', 'M$_{I}$/N$_{3}$', "M$_{I}$/M$_{II'}$",
                      'M$_{S}$/N$_{1}$', 'N$_{1}$/N$_{2}$', 'N$_{1}$/M$_{T}$', 'N$_{1}$/N$_{3}$', "N$_{1}$/M$_{II'}$",
                      'M$_{S}$/N$_{2}$', 'M$_{S}$/M$_{T}$', 'M$_{S}$/N$_{3}$', "M$_{S}$/M$_{II'}$",
                      'N$_{2}$/M$_{T}$', 'N$_{2}$/N$_{3}$', "N$_{2}$/M$_{II'}$",
                      'M$_{T}$/N$_{3}$', "M$_{T}$/M$_{II'}$",
                      "N$_{3}$/M$_{II'}$"]
        theta = np.linspace(start=0, stop=2 * np.pi, num=len(self.averD), endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))
        averD = copy.deepcopy(self.averD)
        averD = np.append(averD, averD[0])
        for sample in secr_intensities:
            sample.append(sample[0])
        for sample in nesecr_intensities:
            sample.append(sample[0])
        errors = copy.deepcopy(self.errors)
        errors = np.append(errors, errors[0])

        colors = {'D': "green", 'M': "red", 'N': "blue", 'O': "black", 'B': "orange", 'U': "purple", }
        linestyles = {'D': '-', 'M': '--', 'N': '-.', 'O': ':', 'B': 'solid', 'U': 'dashed', }
        hatches = {'D': None, 'M': '.', 'N': '/', 'O': '\\', 'B': 'o', 'U': '*', }

        self.RatioWidget.canvas.ax.plot(theta, averD, linewidth=2, linestyle=linestyles['D'],
                                        color=colors["D"])
        self.RatioWidget.canvas.ax.bar(theta, averD, linewidth=0, yerr=errors,
                                       capsize=0.00008,
                                       color=colors["D"], fill=None, ecolor=colors["D"], alpha=0.8)
        flag = str()
        for i in self.Patients:
            try:
                errors_M = copy.deepcopy(self.errors_M)
                errors_N = copy.deepcopy(self.errors_N)
                errors_M = np.append(errors_M, errors_M[0])
                errors_N = np.append(errors_N, errors_N[0])
                for j in range(len(secr_filenames)):
                    if secr_filenames[j][1:] == str(i):
                        self.RatioWidget.canvas.ax.plot(theta, secr_intensities[j], linewidth=2, linestyle=linestyles['M'],
                                       color=colors["M"])
                        self.RatioWidget.canvas.ax.bar(theta, secr_intensities[j], linewidth=0, yerr=errors_M,
                                                       capsize=0.00008, color=colors["M"], fill=None,
                                                       ecolor=colors["M"], alpha=0.8)
                        flag = "M"
                for j in range(len(nesecr_filenames)):
                    if nesecr_filenames[j][1:] == str(i):
                        self.RatioWidget.canvas.ax.plot(theta, nesecr_intensities[j], linewidth=2, linestyle=linestyles['N'],
                                       color=colors["N"])
                        self.RatioWidget.canvas.ax.bar(theta, nesecr_intensities[j], linewidth=0, yerr=errors_N,
                                      capsize=0.00008, color=colors["N"], fill=None, ecolor=colors["N"], alpha=0.8)
                        flag = "N"
            except:
                pass

        _ran = [*range(0, 360, math.floor(360 / len(labels)))]
        if len(_ran) > len(labels):
            _ran.pop(-1)
        self.RatioWidget.canvas.ax.set_thetagrids(_ran, labels, fontsize=16)
        plt.yticks(np.arange(0, 1.5, 0.2), fontsize=8)
        self.RatioWidget.canvas.ax.set(facecolor='#f3f3f3')
        self.RatioWidget.canvas.ax.set_theta_offset(np.pi / 2)
        pl = self.RatioWidget.canvas.ax.yaxis.get_gridlines()
        for line in pl:
            line.get_path()._interpolation_steps = 5

        _ran = [*range(0, 360, math.floor(360 / len(labels)))]
        if len(_ran) > len(labels):
            _ran.pop(-1)
        self.RatioWidget.canvas.ax.set_thetagrids(_ran, labels, fontsize=14)
        self.RatioWidget.canvas.ax.tick_params(pad=10)
        plt.yticks(np.arange(0, 1.5, 0.2), fontsize=10)

        self.RatioWidget.canvas.ax.set_theta_offset(np.pi / 2)
        self.RatioWidget.canvas.ax.set_theta_direction(-1)
        pl = self.RatioWidget.canvas.ax.yaxis.get_gridlines()
        for line in pl:
            line.get_path()._interpolation_steps = 5

        def correct_errorbar(ax, barlen=0.05, errorline=1, color='black'):
            x, y = ax.lines[errorline].get_data()
            del ax.lines[errorline]
            for i in range(len(y)):
                r = sqrt(barlen * barlen / 4 + y[i] * y[i])
                dt = acos((y[i]) / (r))
                newline = Line2D([x[i] - dt, x[i] + dt], [r, r], color=color, linewidth=2, zorder=50 + errorline,
                                 alpha=0.8)
                ax.add_line(newline)

        _indexes = [8,7,5,4,2,1]
        _colors = [colors['M'], colors['M'], colors['N'], colors['N'], colors['D'], colors['D'], ]
        try:
            for i in range(len(_indexes)):
                if len(self.Patients) == 2:
                    _indexes = [8,7,5,4,2,1]
                    correct_errorbar(self.RatioWidget.canvas.ax, barlen=0.1, errorline=_indexes[i], color=_colors[i])
                else:
                    if flag == "M":
                        _colors = [colors['M'], colors['M'], colors['D'], colors['D'], ]
                    else:
                        _colors = [colors['N'], colors['N'], colors['D'], colors['D'], ]
                    _indexes = [5, 4, 2, 1]
                    correct_errorbar(self.RatioWidget.canvas.ax, barlen=0.1, errorline=_indexes[i], color=_colors[i])
        except Exception as error:
            print(error)

        g1 = self.waveD
        g2 = pat_wave
        g3 = non_wave
        cat_par = ['M$_{I}$', 'N$_{1}$', 'M$_{S}$', 'N$_{2}$', 'M$_{T}$', 'N$_{3}$', "M$_{II'}$"]
        width = 0.3
        error_d = np.array(self.w_errors).T
        bottoms = [9, 5, 5, 2, 2, 2, 6.5]

        for index in range(len(g1)):
            self.WaveWidget.canvas.axs[index].bar(1 - width, g1[index] - (g1[index]-bottoms[index]), width=0.3,
                                                  bottom=g1[index]-bottoms[index],
                                                  yerr=error_d[index], ecolor="black", alpha=0.6, color='darkgreen',
                                                  edgecolor="blue", linewidth=0.1, capsize=4, hatch=hatches["D"],)
            for i in self.Patients:
                for j in range(len(secr_filenames)):
                    if secr_filenames[j][1:] == str(i):
                        self.WaveWidget.canvas.axs[index].bar(1 + 1* int(self.Patients.index(i)),
                                                          g2[j][index] - (g1[index]-bottoms[index]),
                                                          width=0.3, bottom=g1[index]-bottoms[index], yerr=self.w_errors_M[index],
                                                          ecolor="black", alpha=0.6, color='red', edgecolor="blue",
                                                          linewidth=0.1, capsize=4, hatch=hatches['M'],)
                # self.WaveWidget.canvas.axs[index].yaxis.set_major_locator(MaxNLocator(integer=True))
                for j in range(len(nesecr_filenames)):
                    if nesecr_filenames[j][1:] == str(i):
                        self.WaveWidget.canvas.axs[index].bar(1 + width * int(self.Patients.index(i)),
                                                          g3[j][index] - (g1[index]-bottoms[index]), width=0.3,
                                                          bottom=g1[index]-bottoms[index], yerr=self.w_errors_N[index],
                                                          ecolor="black", alpha=0.6, color='b',
                                                          edgecolor="mediumblue", linewidth=0.1, capsize=4, hatch=hatches['N'],)
            self.WaveWidget.canvas.axs[index].set_title(fontsize=10, label=cat_par[index])

        self.WaveWidget.canvas.draw()
        self.RatioWidget.canvas.draw()


class MplCanvasPatient(Canvas):
    def __init__(self, type_of_graph, numb):
        if type_of_graph == 'polar':
            dpi = 100
            self.fig = Figure(figsize=(780/dpi, 720/dpi), dpi=dpi)
            self.ax = self.fig.add_subplot(111, projection='polar')
        elif type_of_graph == 'errorbar':
            self.fig, self.axs = plt.subplots(1, numb, figsize=(640 / 100, 620 / 100), constrained_layout=True)
            for i in range(len(self.axs)):
                self.axs[i].xaxis.set_visible(False)
                self.axs[i].yaxis.set_visible(True)
                self.axs[i].tick_params(labelsize=8, direction='in')
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class RatioWidgetPatient(QtWidgets.QWidget):
    def __init__(self, numb, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvasPatient('polar', numb)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class WaveWidgetPatient(QtWidgets.QWidget):
    def __init__(self, numb, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvasPatient('errorbar', numb)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
