from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


class PatWin(QDialog):
    def __init__(self, SpeAn):
        self.SpeAn = SpeAn
        super(PatWin, self).__init__()
        loadUi("C:\\PCA_with_R\\PatientWindow.ui", self)
        self.RatioWidget = RatioWidgetPatient(self.RatioWidget)
        self.WaveWidget = WaveWidgetPatient(self.WaveWidget)
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
            self.Patients = [1,2,3]

        secr_intensities, nesecr_intensities, error_radial, secr_waves, nesecr_waves = \
            self.SpeAn.show_patient_graph(self.Patients)

        labels = ['A1/M1', 'A1/A2', 'A1/M2', 'A1/A3', 'M1/A2', 'M1/M2', 'M1/A3', 'A2/M2', 'A2/A3', 'M2/A3']
        theta = np.linspace(start=0, stop=2 * np.pi, num=len(self.SpeAn.copy_result_d), endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))
        self.SpeAn.copy_result_d = np.append(self.SpeAn.copy_result_d, self.SpeAn.copy_result_d[0])
        for sample in secr_intensities:
            sample.append(sample[0])
        for sample in nesecr_intensities:
            sample.append(sample[0])
        error_radial = np.append(error_radial, error_radial[0])
        for sample in secr_intensities:
            self.RatioWidget.canvas.ax.errorbar(theta, sample, linewidth=2, xerr=0, yerr=0, color="red")
        self.RatioWidget.canvas.ax.errorbar(theta, self.SpeAn.copy_result_d, linewidth=2, xerr=0, yerr=error_radial,
                                            color="darkgreen", ecolor='black')
        for sample in nesecr_intensities:
            self.RatioWidget.canvas.ax.errorbar(theta, sample, linewidth=2, xerr=0, yerr=0, color="mediumblue")
        self.RatioWidget.canvas.ax.set_thetagrids(range(0, 360, int(360 / len(labels))), labels)
        plt.yticks(np.arange(0, 1.5, 0.2), fontsize=8)
        self.RatioWidget.canvas.ax.set(facecolor='#f3f3f3')
        self.RatioWidget.canvas.ax.set_theta_offset(np.pi / 2)
        pl = self.RatioWidget.canvas.ax.yaxis.get_gridlines()
        for line in pl:
            line.get_path()._interpolation_steps = 5

        g1 = self.SpeAn.result_waves_d
        g2 = secr_waves
        g3 = nesecr_waves
        cat_par = ['Amide-I', 'Min 1-2', 'Amide-II', 'Min 2-3', 'Amide-III']
        width = 0.3
        error_d = np.array([0.89, 0.364, 0.625, 0.483, 0.246]).T
        error_p = np.array([0.1, 0.3, 0.2, 0.4, 0.5]).T
        bottom = [1638.5, 1595.5, 1569.5, 1503.5, 1448.5]
        for index in range(len(g1)):
            self.WaveWidget.canvas.axs[index].bar(1 - width, g1[index] - bottom[index], width=0.3,
                                                  bottom=bottom[index],
                                                  yerr=error_d[index], ecolor="black", alpha=0.6, color='darkgreen',
                                                  edgecolor="blue", linewidth=0.1)
            for sample in g2:
                self.WaveWidget.canvas.axs[index].bar(1 + width * int(g2.index(sample)),
                                                      sample[index] - bottom[index],
                                                      width=0.3, bottom=bottom[index], yerr=error_p[index],
                                                      ecolor="black", alpha=0.6, color='red', edgecolor="blue",
                                                      linewidth=0.1)
                self.WaveWidget.canvas.axs[index].yaxis.set_major_locator(MaxNLocator(integer=True))
            for sample in g3:
                self.WaveWidget.canvas.axs[index].bar(1 + width * (len(g2) + int(g3.index(sample))),
                                                      sample[index] - bottom[index], width=0.3,
                                                      bottom=bottom[index],
                                                      yerr=error_p[index], ecolor="black", alpha=0.6, color='b',
                                                      edgecolor="mediumblue", linewidth=0.1)
            self.WaveWidget.canvas.axs[index].set_title(fontsize=10, label=cat_par[index])

        self.WaveWidget.canvas.draw()
        self.RatioWidget.canvas.draw()


class MplCanvasPatient(Canvas):
    def __init__(self, type_of_graph):
        if type_of_graph == 'polar':
            dpi = 100
            self.fig = Figure(figsize=(780/dpi, 720/dpi), dpi=dpi)
            self.ax = self.fig.add_subplot(111, projection='polar')
        elif type_of_graph == 'errorbar':
            self.fig, self.axs = plt.subplots(1, 5, figsize=(780 / 100, 720 / 100), constrained_layout=True)
            for i in range(len(self.axs)):
                self.axs[i].xaxis.set_visible(False)
                self.axs[i].yaxis.set_visible(True)
                self.axs[i].tick_params(labelsize=8, direction='in')
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class RatioWidgetPatient(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvasPatient('polar')
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class WaveWidgetPatient(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvasPatient('errorbar')
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)