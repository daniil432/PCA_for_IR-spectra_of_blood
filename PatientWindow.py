from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


class Ui_PatientWindow(object):
    def Signal(self, main):
        self.main = main


    def Home(self):
        self.PatientWindow.close()
        self.SecondWindow.show()


    def showGraph(self):
        self.RatioWindow.canvas.ax.clear()
        self.WaveWindow.canvas.axs[0].clear()
        self.WaveWindow.canvas.axs[1].clear()
        self.WaveWindow.canvas.axs[2].clear()
        self.WaveWindow.canvas.axs[3].clear()
        self.WaveWindow.canvas.axs[4].clear()

        Patients_temp = self.Patient_int.toPlainText()
        Patients_temp = Patients_temp.replace(' ', '')
        Patients_temp = Patients_temp.replace('.', ',')
        input_temp = Patients_temp.split(',')

        self.Patients = []
        for j in input_temp:
            self.Patients.append(int(j))
        if self.Patients == []:
            pass

        secr_intensities, nesecr_intensities, error_radial, secr_waves, nesecr_waves = \
            self.main.show_patient_graph(self.Patients)

        labels = ['A1/M1', 'A1/A2', 'A1/M2', 'A1/A3', 'M1/A2', 'M1/M2', 'M1/A3', 'A2/M2', 'A2/A3', 'M2/A3']
        theta = np.linspace(start=0, stop=2 * np.pi, num=len(self.main.copy_result_d), endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))
        self.main.copy_result_d = np.append(self.main.copy_result_d, self.main.copy_result_d[0])
        for sample in secr_intensities:
            sample.append(sample[0])
        for sample in nesecr_intensities:
            sample.append(sample[0])
        error_radial = np.append(error_radial, error_radial[0])
        for sample in secr_intensities:
            self.RatioWindow.canvas.ax.errorbar(theta, sample, linewidth=2, xerr=0, yerr=0, color="red")
        self.RatioWindow.canvas.ax.errorbar(theta, self.main.copy_result_d, linewidth=2, xerr=0, yerr=error_radial,
                                            color="darkgreen", ecolor='black')
        for sample in nesecr_intensities:
            self.RatioWindow.canvas.ax.errorbar(theta, sample, linewidth=2, xerr=0, yerr=0, color="mediumblue")
        self.RatioWindow.canvas.ax.set_thetagrids(range(0, 360, int(360 / len(labels))), labels)
        plt.yticks(np.arange(0, 1.5, 0.2), fontsize=8)
        self.RatioWindow.canvas.ax.set(facecolor='#f3f3f3')
        self.RatioWindow.canvas.ax.set_theta_offset(np.pi / 2)
        pl = self.RatioWindow.canvas.ax.yaxis.get_gridlines()
        for line in pl:
            line.get_path()._interpolation_steps = 5

        g1 = self.main.result_waves_d
        g2 = secr_waves
        g3 = nesecr_waves
        cat_par = ['Amide-I', 'Min 1-2', 'Amide-II', 'Min 2-3', 'Amide-III']
        width = 0.3
        error_d = np.array([0.89, 0.364, 0.625, 0.483, 0.246]).T
        error_p = np.array([0.1, 0.3, 0.2, 0.4, 0.5]).T
        bottom = [1638.5, 1595.5, 1569.5, 1503.5, 1448.5]
        for index in range(len(g1)):
            self.WaveWindow.canvas.axs[index].bar(1 - width, g1[index] - bottom[index], width=0.3, bottom=bottom[index],
                                                  yerr=error_d[index], ecolor="black", alpha=0.6, color='darkgreen',
                                                  edgecolor="blue", linewidth=0.1)
            for sample in g2:
                self.WaveWindow.canvas.axs[index].bar(1 + width * int(g2.index(sample)), sample[index] - bottom[index],
                                                      width=0.3, bottom=bottom[index], yerr=error_p[index],
                                                      ecolor="black", alpha=0.6, color='red', edgecolor="blue",
                                                      linewidth=0.1)
                self.WaveWindow.canvas.axs[index].yaxis.set_major_locator(MaxNLocator(integer=True))
            for sample in g3:
                self.WaveWindow.canvas.axs[index].bar(1 + width * (len(g2) + int(g3.index(sample))),
                                                      sample[index] - bottom[index], width=0.3, bottom=bottom[index],
                                                      yerr=error_p[index], ecolor="black", alpha=0.6, color='b',
                                                      edgecolor="mediumblue", linewidth=0.1)
            self.WaveWindow.canvas.axs[index].set_title(fontsize=10, label=cat_par[index])

        self.WaveWindow.canvas.draw()
        self.RatioWindow.canvas.draw()


    def setupUi(self, PatientWindow, SecondWindow):
        self.PatientWindow = PatientWindow
        self.SecondWindow = SecondWindow
        PatientWindow.setObjectName("PatientWindow")
        PatientWindow.resize(1600, 900)
        self.centralwidget = QtWidgets.QWidget(PatientWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setEnabled(True)
        self.label_3.setGeometry(QtCore.QRect(630, 0, 331, 60))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        self.label_3.setFont(font)
        self.label_3.setMouseTracking(True)
        self.label_3.setTabletTracking(False)
        self.label_3.setAcceptDrops(False)
        self.label_3.setAutoFillBackground(False)
        self.label_3.setScaledContents(False)
        self.label_3.setObjectName("label_3")
        self.Patient_int = QtWidgets.QTextEdit(self.centralwidget)
        self.Patient_int.setGeometry(QtCore.QRect(630, 60, 330, 40))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        self.Patient_int.setFont(font)
        self.Patient_int.setObjectName("Patient_int")
        self.AcceptButton = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.showGraph())
        self.AcceptButton.setGeometry(QtCore.QRect(760, 110, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.AcceptButton.setFont(font)
        self.AcceptButton.setObjectName("AcceptButton")
        self.CloseButton = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.Home())
        self.CloseButton.setGeometry(QtCore.QRect(760, 850, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.CloseButton.setFont(font)
        self.CloseButton.setObjectName("CloseButton")
        self.WaveWindow = WaveWidget(self.centralwidget)
        self.WaveWindow.setGeometry(QtCore.QRect(799, 139, 781, 721))
        self.WaveWindow.setObjectName("WaveWindow")
        self.RatioWindow = RatioWidget(self.centralwidget)
        self.RatioWindow.setGeometry(QtCore.QRect(20, 139, 781, 721))
        self.RatioWindow.setObjectName("RatioWindow")
        PatientWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(PatientWindow)
        self.statusbar.setObjectName("statusbar")
        PatientWindow.setStatusBar(self.statusbar)

        self.retranslateUi(PatientWindow)
        QtCore.QMetaObject.connectSlotsByName(PatientWindow)

    def retranslateUi(self, PatientWindow):
        _translate = QtCore.QCoreApplication.translate
        PatientWindow.setWindowTitle(_translate("PatientWindow", "MainWindow"))
        self.label_3.setText(_translate("PatientWindow", "<html><head/><body><p align=\"center\">Введите номера "
                                                         "пациентов:</p></body></html>"))
        self.Patient_int.setHtml(_translate("PatientWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" "
                                                             "\"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                             "<html><head><meta name=\"qrichtext\" content=\"1\""
                                                             " /><style type=\"text/css\">\n"
                                                             "p, li { white-space: pre-wrap; }\n"
                                                             "</style></head><body style=\" font-family:\'Arial\'; "
                                                             "font-size:18pt; font-weight:400; font-style:normal;\">"
                                                             "\n""<p align=\"center\" style=\"-qt-paragraph-type:empty;"
                                                             "margin-top:0px; margin-bottom:0px; margin-left:0px; "
                                                             "margin-right:0px; -qt-block-indent:0; text-indent:0px; "
                                                             "font-family:\'MS Shell Dlg 2\'; "
                                                             "font-size:8.25pt;\"><br /></p></body></html>"))
        self.AcceptButton.setText(_translate("PatientWindow", "ok"))
        self.CloseButton.setText(_translate("PatientWindow", "Назад"))


class MplCanvas(Canvas):
    def __init__(self, type_of_graph):
        if type_of_graph == 'polar':
            self.fig = Figure(figsize=(14, 14), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='polar')
        elif type_of_graph == 'errorbar':
            self.fig, self.axs = plt.subplots(1, 5, figsize=(14, 14), constrained_layout=True)
            for i in range(len(self.axs)):
                self.axs[i].xaxis.set_visible(False)
                self.axs[i].yaxis.set_visible(True)
                self.axs[i].tick_params(labelsize=8, direction='in')
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class RatioWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas('polar')
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class WaveWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas('errorbar')
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)

