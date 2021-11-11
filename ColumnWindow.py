from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class Ui_ColumnWindow(object):
    def Signal(self, main, signal, t_pca, p_pca, t_der1, p_der1, t_der2, p_der2):
        self.main = main
        self.signal = signal
        self.t_matrix_pca = t_pca
        self.p_matrix_pca = p_pca
        self.t_matrix_der1 = t_der1
        self.p_matrix_der1 = p_der1
        self.t_matrix_der2 = t_der2
        self.p_matrix_der2 = p_der2


    def radioButtonChecking(self):
        if self.SpectraButton.isChecked():
            self.t_matrix = self.t_matrix_pca
            self.p_matrix = self.p_matrix_pca
        elif self.Derivative_1_Button.isChecked():
            self.t_matrix = self.t_matrix_der1
            self.p_matrix = self.p_matrix_der1
        elif self.Derivative_2_Button.isChecked():
            self.t_matrix = self.t_matrix_der2
            self.p_matrix = self.p_matrix_der2


    def Home(self):
        self.ColumnWindow.close()
        self.SecondWindow.show()


    def showGraph(self):
        self.radioButtonChecking()
        self.filenames = self.main.filenames
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
                self.plotScores()
            elif self.signal == 2:
                self.plotLoadings()
            elif self.signal == 3:
                self.plot3D()


    def plotScores(self):
        self.colGraph.canvas.ax.clear()
        first_column = self.Columns[0]
        second_column = self.Columns[1]
        first_column -= 1
        second_column -= 1
        self.x = self.t_matrix[:, first_column]
        self.y = self.t_matrix[:, second_column]
        for index in range(len(self.filenames)):
            if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="red", marker="o", s=50)
                self.colGraph.canvas.ax.annotate(self.filenames[index], (self.x[index], self.y[index]))
            elif self.filenames[index][0] == 'N':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="blue", marker="o", s=50)
                self.colGraph.canvas.ax.annotate(self.filenames[index], (self.x[index], self.y[index]))
            elif self.filenames[index][0] == 'D':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="green", marker="o", s=50)
                self.colGraph.canvas.ax.annotate(self.filenames[index], (self.x[index], self.y[index]))
            elif (self.filenames[index][0] == 'O') or (self.filenames[index][0] == 'B'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], color="black", marker="o", s=50)
                # plt.annotate(filenames[index], (x[index], y[index]))
        self.colGraph.canvas.draw()


    def plotLoadings(self):
        first_column = self.Columns[0]
        second_column = self.Columns[1]
        first_column -= 1
        second_column -= 1
        self.x = self.p_matrix[:, first_column]
        self.y = self.p_matrix[:, second_column]
        self.colGraph.canvas.ax.clear()
        self.colGraph.canvas.ax.scatter(self.x, self.y, color="black", marker="o", s=12)
        self.colGraph.canvas.draw()


    def plot3D(self):
        self.colGraph.canvas.ax.clear()
        first_column = self.Columns[0]
        second_column = self.Columns[1]
        third_column = self.Columns[2]
        first_column -= 1
        second_column -= 1
        third_column -= 1
        self.x = self.t_matrix[:, first_column]
        self.y = self.t_matrix[:, second_column]
        self.z = self.t_matrix[:, third_column]
        for index in range(len(self.filenames)):
            if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="red")
            elif self.filenames[index][0] == 'N':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="blue")
            elif self.filenames[index][0] == 'D':
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="green")
            elif (self.filenames[index][0] == 'O') or (self.filenames[index][0] == 'B'):
                self.colGraph.canvas.ax.scatter(self.x[index], self.y[index], self.z[index], color="black")
        """for angle in range(0, 360):
            self.colGraph.canvas.ax.view_init(0, angle)"""
        self.colGraph.canvas.draw()


    def setupUi(self, ColumnWindow, SecondWindow):
        self.ColumnWindow = ColumnWindow
        self.SecondWindow = SecondWindow
        ColumnWindow.setObjectName("ColumnWindow")
        ColumnWindow.resize(1280, 858)
        self.centralwidget = QtWidgets.QWidget(ColumnWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Columns_int = QtWidgets.QTextEdit(self.centralwidget)
        self.Columns_int.setGeometry(QtCore.QRect(470, 70, 340, 40))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.Columns_int.setFont(font)
        self.Columns_int.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Columns_int.setObjectName("Columns_int")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setEnabled(True)
        self.label_3.setGeometry(QtCore.QRect(470, 20, 341, 41))
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
        self.pushButton = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.showGraph())
        self.pushButton.setGeometry(QtCore.QRect(600, 160, 80, 25))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.colGraph = MplWidget(self.signal, self.centralwidget)
        self.colGraph.setGeometry(QtCore.QRect(20, 200, 1240, 601))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.colGraph.setFont(font)
        self.colGraph.setObjectName("colGraph")
        self.SpectraButton = QtWidgets.QRadioButton(self.centralwidget)
        self.SpectraButton.setGeometry(QtCore.QRect(320, 120, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.SpectraButton.setFont(font)
        self.SpectraButton.setChecked(True)
        self.SpectraButton.setObjectName("SpectraButton")
        self.Derivative_1_Button = QtWidgets.QRadioButton(self.centralwidget)
        self.Derivative_1_Button.setGeometry(QtCore.QRect(540, 120, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.Derivative_1_Button.setFont(font)
        self.Derivative_1_Button.setObjectName("Derivative_1_Button")
        self.Derivative_2_Button = QtWidgets.QRadioButton(self.centralwidget)
        self.Derivative_2_Button.setGeometry(QtCore.QRect(770, 120, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.Derivative_2_Button.setFont(font)
        self.Derivative_2_Button.setObjectName("Derivative_2_Button")
        self.CloseButton = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.Home())
        self.CloseButton.setGeometry(QtCore.QRect(600, 810, 80, 25))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.CloseButton.setFont(font)
        self.CloseButton.setObjectName("CloseButton")
        ColumnWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(ColumnWindow)
        self.statusbar.setObjectName("statusbar")
        ColumnWindow.setStatusBar(self.statusbar)

        self.retranslateUi(ColumnWindow)
        QtCore.QMetaObject.connectSlotsByName(ColumnWindow)


    def retranslateUi(self, ColumnWindow):
        _translate = QtCore.QCoreApplication.translate
        ColumnWindow.setWindowTitle(_translate("ColumnWindow", "MainWindow"))
        self.label_3.setText(_translate("ColumnWindow", "<html><head/><body><p align=\"center\">"
                                                        "Введите номера столбцов:</p></body></html>"))
        self.pushButton.setText(_translate("ColumnWindow", "Принять"))
        self.SpectraButton.setText(_translate("ColumnWindow", "Для самих спектров"))
        self.Derivative_1_Button.setText(_translate("ColumnWindow", "Для 1-й производной"))
        self.Derivative_2_Button.setText(_translate("ColumnWindow", "Для 2-й производной"))
        self.CloseButton.setText(_translate("ColumnWindow", "Назад"))


class MplCanvas(Canvas):
    def __init__(self, signal):
        self.fig = Figure(figsize=(16, 16), dpi=100)
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
