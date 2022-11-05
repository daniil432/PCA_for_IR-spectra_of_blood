import math

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class PatWin(QDialog):
    def __init__(self, ratio, waves, ratio_waves, filenames):
        self.ratio = ratio
        self.waves = waves
        self.averD = ratio_waves['D'][0]
        self.waveD = ratio_waves['D'][1]
        self.errors = ratio_waves['D'][2]
        self.w_errors = ratio_waves['D'][3]
        self.filenames = filenames
        super(PatWin, self).__init__()
        loadUi("C:\\PCA_with_R\\PatientWindow.ui", self)
        self.RatioWidget = RatioWidgetPatient(len(self.waveD), self.RatioWidget)
        self.WaveWidget = WaveWidgetPatient(len(self.waveD), self.WaveWidget)
        self.AcceptButton.clicked.connect(self.showGraph)
        self.CloseButton.clicked.connect(self.close)

    def showGraph(self):
        self.RatioWidget.canvas.ax.clear()
        self.WaveWidget.canvas.axs[0].clear()
        self.WaveWidget.canvas.axs[1].clear()
        self.WaveWidget.canvas.axs[2].clear()
        self.WaveWidget.canvas.axs[3].clear()
        self.WaveWidget.canvas.axs[4].clear()

        if self.Patient_int.toPlainText() != '':
            Patients_temp = self.Patient_int.toPlainText()
            Patients_temp = Patients_temp.replace(' ', '')
            Patients_temp = Patients_temp.replace('.', ',')
            input_temp = Patients_temp.split(',')
            self.Patients = []
            for j in input_temp:
                self.Patients.append(int(j))
            if self.Patients == []:
                pass
        else:
            self.Patients = [1, 2, 3]

        secr_intensities = []
        nesecr_intensities = []
        pat_wave = []
        non_wave = []
        for name in range(len(self.filenames)):
            if self.filenames[name][0] == 'M' or self.filenames[name][0] == 'S':
                secr_intensities.append(self.ratio[name])
                pat_wave.append(self.waves[name])
            elif self.filenames[name][0] == 'N':
                nesecr_intensities.append(self.ratio[name])
                non_wave.append(self.waves[name])
        labels = ['A1/Min1', 'A1/s.ch.', 'A1/Min2', 'A1/Tyr', 'A1/Min3', "A1/A2'",
                  'Min1/s.ch.', 'Min1/Min2', 'Min1/Tyr', 'Min1/Min3', "Min1/A2'",
                  's.ch./Min2', 's.ch./Tyr', 's.ch./Min3', "s.ch./A2'",
                  'Min2/Tyr', 'Min2/Min3', "Min2/A2'",
                  'Tyr/Min3', "Tyr/A2'",
                  "Min3/A2'"]
        theta = np.linspace(start=0, stop=2 * np.pi, num=len(self.averD), endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))
        self.averD = np.append(self.averD, self.averD[0])
        for sample in secr_intensities:
            sample.append(sample[0])
        for sample in nesecr_intensities:
            sample.append(sample[0])
        self.errors = np.append(self.errors, self.errors[0])
        self.RatioWidget.canvas.ax.errorbar(theta, self.averD, linewidth=2, xerr=0, yerr=self.errors,
                                            color="darkgreen", ecolor='black')
        for sample in secr_intensities:
            self.RatioWidget.canvas.ax.errorbar(theta, sample, linewidth=2, xerr=0, yerr=0, color="red")
        for sample in nesecr_intensities:
            self.RatioWidget.canvas.ax.errorbar(theta, sample, linewidth=2, xerr=0, yerr=0, color="mediumblue")
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

        g1 = self.waveD
        g2 = pat_wave
        g3 = non_wave
        cat_par = ['Amide-I', 'Min 1', 'Side chains', 'Min2', 'Tyr', 'Min 3', "Amide-II'"]
        width = 0.3
        error_d = np.array(self.w_errors).T
        for index in range(len(g1)):
            self.WaveWidget.canvas.axs[index].bar(1 - width, g1[index] - min(g1[index], g2[0][index]), width=0.3,
                                                  bottom=min(g1[index], g2[0][index]),
                                                  yerr=error_d[index], ecolor="black", alpha=0.6, color='darkgreen',
                                                  edgecolor="blue", linewidth=0.1)
            for sample in g2:
                self.WaveWidget.canvas.axs[index].bar(1 + width * int(g2.index(sample)),
                                                      sample[index] - min(g1[index], g2[0][index]),
                                                      width=0.3, bottom=min(g1[index], g2[0][index]),
                                                      ecolor="black", alpha=0.6, color='red', edgecolor="blue",
                                                      linewidth=0.1)
                self.WaveWidget.canvas.axs[index].yaxis.set_major_locator(MaxNLocator(integer=True))
            for sample in g3:
                self.WaveWidget.canvas.axs[index].bar(1 + width * (len(g2) + int(g3.index(sample))),
                                                      sample[index] - min(g1[index], g2[0][index]), width=0.3,
                                                      bottom=min(g1[index], g2[0][index]),
                                                      ecolor="black", alpha=0.6, color='b',
                                                      edgecolor="mediumblue", linewidth=0.1)
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
            self.fig, self.axs = plt.subplots(1, numb, figsize=(780 / 100, 720 / 100), constrained_layout=True)
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
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class WaveWidgetPatient(QtWidgets.QWidget):
    def __init__(self, numb, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvasPatient('errorbar', numb)
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
