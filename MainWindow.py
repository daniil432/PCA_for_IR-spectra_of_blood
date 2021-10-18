from PyQt5 import QtCore, QtGui, QtWidgets
from SecondWindow import Ui_SecondWindow
from main import Spectra_Anal


class Ui_MainWindow(object):
    def openDpt(self):
        SecondWindow = QtWidgets.QMainWindow()
        ui = Ui_SecondWindow()
        ui.setupUi(SecondWindow, MainWindow)
        MainWindow.hide()
        SecondWindow.show()


    def openCsv(self):
        self.main = Spectra_Anal()
        SecondWindow = QtWidgets.QMainWindow()
        ui = Ui_SecondWindow()
        ui.setupUi(SecondWindow, MainWindow)
        self.main.read_eigenvalues_and_eigenvectors_from_files()
        self.main.calculate_t_and_p_matrix()
        ui.Signal_Csv(self.main)
        MainWindow.hide()
        SecondWindow.show()


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(515, 380)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 30, 401, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(26)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.button_csv = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.openCsv())
        self.button_csv.setGeometry(QtCore.QRect(20, 180, 401, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.button_csv.setFont(font)
        self.button_csv.setObjectName("button_csv")
        self.button_dpt = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.openDpt())
        self.button_dpt.setGeometry(QtCore.QRect(20, 110, 471, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.button_dpt.setFont(font)
        self.button_dpt.setObjectName("button_dpt")
        self.directory_csv = QtWidgets.QToolButton(self.centralwidget)
        self.directory_csv.setGeometry(QtCore.QRect(430, 180, 61, 61))
        self.directory_csv.setObjectName("directory_csv")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 280, 391, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 515, 21))
        self.menubar.setObjectName("menubar")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout_us = QtWidgets.QAction(MainWindow)
        self.actionAbout_us.setObjectName("actionAbout_us")
        self.menuHelp.addAction(self.actionAbout_us)
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Что вы хотите сделать?"))
        self.button_csv.setText(_translate("MainWindow", "Воспроизвести готовое исследование"))
        self.button_dpt.setText(_translate("MainWindow", "Прочитать новые спектры из .dpt"))
        self.directory_csv.setText(_translate("MainWindow", "..."))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">"
                                                      "Это начальная страница</p></body></html>"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionAbout_us.setText(_translate("MainWindow", "About us"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
