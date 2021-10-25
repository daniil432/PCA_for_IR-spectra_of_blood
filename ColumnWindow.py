from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class Ui_ColumnWindow(object):
    def Signal(self, main, signal):
        self.main = main
        self.signal = signal


    def Home(self):
        self.ColumnWindow.close()
        self.SecondWindow.show()


    def showGraph(self):
        Columns_temp = self.Columns_int.toPlainText()
        Columns_temp = Columns_temp.replace(' ', '')
        Columns_temp = Columns_temp.replace('.', ',')
        input_temp = Columns_temp.split(',')
        self.Columns = []
        for j in input_temp:
            self.Columns.append(int(j))
        if self.Columns == []:
            pass
        else:
            if self.signal == 1:
                self.x, self.y, self.filenames = self.main.show_graphic_of_t_matrix(Columns=self.Columns)
                self.plotScores()
            elif self.signal == 2:
                self.x, self.y = self.main.show_graphic_of_p_matrix(Columns=self.Columns)
                self.plotLoadings()
            elif self.signal == 3:
                self.x, self.y, self.z, self.filenames = self.main.show_graphic_3D(Columns=self.Columns)
                self.plot3D()


    def plotScores(self):
        self.colGraph.canvas.ax.clear()
        for index in range(len(self.filenames)):
            if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="red", marker="o", s=50)
                # plt.annotate(filenames[index], (x[index], y[index]))
            elif self.filenames[index][0] == 'N':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="blue", marker="o", s=50)
                # plt.annotate(filenames[index], (x[index], y[index]))
            elif self.filenames[index][0] == 'D':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="green", marker="o", s=50)
                # plt.annotate(filenames[index], (x[index], y[index]))
            elif (self.filenames[index][0] == 'O') or (self.filenames[index][0] == 'B'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="black", marker="o", s=50)
                # plt.annotate(filenames[index], (x[index], y[index]))
        self.colGraph.canvas.draw()


    def plotLoadings(self):
        self.colGraph.canvas.ax.clear()
        self.colGraph.canvas.ax.scatter(self.x, self.y, color="black", marker="o", s=12)
        self.colGraph.canvas.draw()


    def plot3D(self):
        self.colGraph.canvas.ax.clear()
        for index in range(len(self.filenames)):
            if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="red")
            elif self.filenames[index][0] == 'N':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="blue")
            elif self.filenames[index][0] == 'D':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="green")
            elif (self.filenames[index][0] == 'O') or (self.filenames[index][0] == 'B'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="black")
        for angle in range(0, 360):
            self.colGraph.canvas.ax.view_init(0, angle)
        self.colGraph.canvas.draw()




    def setupUi(self, ColumnWindow, SecondWindow):
        self.ColumnWindow = ColumnWindow
        self.SecondWindow = SecondWindow
        ColumnWindow.setObjectName("ColumnWindow")
        ColumnWindow.resize(982, 796)
        self.centralwidget = QtWidgets.QWidget(ColumnWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Columns_int = QtWidgets.QTextEdit(self.centralwidget)
        self.Columns_int.setGeometry(QtCore.QRect(320, 80, 331, 41))
        self.Columns_int.setObjectName("Columns_int")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setEnabled(True)
        self.label_3.setGeometry(QtCore.QRect(320, 20, 331, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setPointSize(18)
        self.label_3.setFont(font)
        self.label_3.setMouseTracking(True)
        self.label_3.setTabletTracking(False)
        self.label_3.setAcceptDrops(False)
        self.label_3.setAutoFillBackground(False)
        self.label_3.setScaledContents(False)
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.showGraph())
        self.pushButton.setGeometry(QtCore.QRect(450, 130, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.colGraph = MplWidget(self.signal, self.centralwidget)
        self.colGraph.setGeometry(QtCore.QRect(20, 170, 941, 601))
        self.colGraph.setObjectName("colGraph")
        ColumnWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(ColumnWindow)
        self.statusbar.setObjectName("statusbar")
        self.CloseButton = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.Home())
        self.CloseButton.setGeometry(QtCore.QRect(450, 690, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.CloseButton.setFont(font)
        self.CloseButton.setObjectName("CloseButton")
        ColumnWindow.setStatusBar(self.statusbar)

        self.retranslateUi(ColumnWindow)
        QtCore.QMetaObject.connectSlotsByName(ColumnWindow)


    def retranslateUi(self, ColumnWindow):
        _translate = QtCore.QCoreApplication.translate
        ColumnWindow.setWindowTitle(_translate("ColumnWindow", "MainWindow"))
        self.label_3.setText(_translate("ColumnWindow", "<html><head/><body><p align=\"center\">"
                                                        "Введите номера столбцов:</p></body></html>"))
        self.pushButton.setText(_translate("ColumnWindow", "ok"))
        self.CloseButton.setText(_translate("PatientWindow", "Назад"))


class MplCanvas(Canvas):
    def __init__(self, signal):
        self.fig = Figure(figsize=(14, 14), dpi=100)
        if signal == 1:
            self.ax = self.fig.add_subplot(111)
        elif signal == 2:
            self.ax = self.fig.add_subplot(111)
        elif signal == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            label.set_fontsize(16)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
        if signal == 3:
            self.ax.mouse_init()
        else:
            pass


class MplWidget(QtWidgets.QWidget):
    def __init__(self, signal, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas(signal=signal)
        self.toolbar = NavigationToolbar(self.canvas, self, True)
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
