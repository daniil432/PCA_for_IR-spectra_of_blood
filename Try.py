from Analization import Spectra_Anal
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from numpy import array
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5 import QtGui, QtCore
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog, QListWidgetItem, \
    QTableWidgetItem



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("MainWindow.ui", self)
        self.show()


class SecondWindow(QMainWindow):
    def __init__(self):
        super(SecondWindow, self).__init__()
        loadUi("SecondWindow.ui", self)
        self.show()


class ColumnWindow(QMainWindow):
    def __init__(self):
        super(ColumnWindow, self).__init__()
        loadUi("ColumnWindow.ui", self)
        self.show()


class AverageWindow(QMainWindow):
    def __init__(self):
        super(AverageWindow, self).__init__()
        loadUi("AverageWindow.ui", self)

        input_data = "1500-1525"
        research_name = "CHLEN"
        path_dpt = 'C:\PCA_with_R\input_dpt'
        normalization = 'y'

        self.main = Spectra_Anal()
        self.main.read_files(input_data=input_data, normalization=normalization, path_dpt=path_dpt)
        self.main.cutting_spectra_and_finding_ratio()
        self.main.sorting_ratio_and_waves_by_names()
        self.main.calculate_ratio()
        self.main.calculate_and_sort_eigenvalues_and_vectors(self.main.input_matrix)
        self.t_matrix_pca, self.p_matrix_pca = self.main.calculate_t_and_p_matrix()
        der1graph = self.main.derivative_function(self.main.all_samples_for_derivative)
        der2graph = self.main.derivative_function(der1graph)
        xd, yd = der2graph[0], der2graph[1]
        plt.figure()
        plt.plot(xd, yd, '-', linewidth=4, label='Data')
        plt.show()
        # self.main.fitting(der2graph, 4)
        der1pca = der1graph[1:]
        der2pca = der2graph[1:]
        self.waves = self.main.all_samples_for_derivative[0]
        self.main.derivative_saving(der2graph)
        self.main.calculate_and_sort_eigenvalues_and_vectors(der1pca)
        self.t_matrix_der1, self.p_matrix_der1 = self.main.calculate_t_and_p_matrix()
        self.main.calculate_and_sort_eigenvalues_and_vectors(der2pca)
        self.t_matrix_der2, self.p_matrix_der2 = self.main.calculate_t_and_p_matrix()
        self.main.write_eigenvalues_and_eigenvectors_in_files(research_name, self.t_matrix_pca, self.p_matrix_pca,
                                                              self.t_matrix_der1, self.p_matrix_der1,
                                                              self.t_matrix_der2, self.p_matrix_der2)
        self.waves_loadings = [self.main.one_wave, der1graph[0], der2graph[0]]
        # self.main.show_graphic_of_eigenvalues_and_pc()

        self.CloseButton.clicked.connect(self.closeAverage)
        "________________________________________________________________________________"
        self.RatioWidget = RatioWidgetAverage(self.main, self.RatioWidget)
        self.WaveWidget = WaveWidgetAverage(self.main, self.WaveWidget)
        self.show()


    def closeAverage(self):
        self.AverageWindow.close()
        self.SecondWindow.show()


class PatientWindow(QMainWindow):
    def __init__(self):
        super(PatientWindow, self).__init__()
        loadUi("PatientWindow.ui", self)
        self.show()


class MplCanvasAverage(Canvas):
    def __init__(self, type_of_graph):
        if type_of_graph == 'polar':
            self.fig = Figure(figsize=(7, 7), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='polar')
        elif type_of_graph == 'errorbar':
            self.fig, self.ax = plt.subplots(1, 5, figsize=(7, 7), constrained_layout=True)
            for i in range(len(self.ax)):
                self.ax[i].xaxis.set_visible(False)
                self.ax[i].yaxis.set_visible(True)
                self.ax[i].tick_params(labelsize=8, direction='in')
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class RatioWidgetAverage(QtWidgets.QWidget):
    def __init__(self, main, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.main = main
        error_radial = [0.5, 0.4, 0.001, 0.07, 0.01, 0.001, 0.001, 0.2, 0.03, 0.001]
        for i in range(len(error_radial)):
            error_radial[i] = error_radial[i] * self.main.normal[i]
        self.result_d = np.append(self.main.result_d, self.main.result_d[0])
        self.result_p = np.append(self.main.result_p, self.main.result_p[0])
        self.result_n = np.append(self.main.result_n, self.main.result_n[0])
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
    def __init__(self, main, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.main = main
        g1 = self.main.result_waves_d
        g2 = self.main.result_waves_p
        g3 = self.main.result_waves_n
        self.canvas = MplCanvasAverage('errorbar')
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = AverageWindow()
    sys.exit(app.exec())


import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
import matplotlib.pyplot as plt
import ColumnWindow
import AverageWindow
import PatientWindow


