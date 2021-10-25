from PyQt5 import QtCore, QtGui, QtWidgets
from ColumnWindow import Ui_ColumnWindow
from PatientWindow import Ui_PatientWindow
from AverageWindow import Ui_AverageWindow
from main import Spectra_Anal
import os
import glob
import matplotlib.pyplot as plt


class Ui_SecondWindow(object):
    def Signal_Csv(self, main):
        self.main = main
        self.main.sorting_ratio_and_waves_by_names()
        self.main.calculate_ratio()
        self.Accept_Button.setEnabled(False)
        self.Diapason_choose.setEnabled(False)
        self.pathText.setEnabled(False)
        self.Research_name.setEnabled(False)
        self.checkBox.setEnabled(False)
        self.directory_dpt.setEnabled(False)
        self.Scores_2D.setEnabled(True)
        self.Scores_3D.setEnabled(True)
        self.Loadings_2D.setEnabled(True)
        self.Average_all.setEnabled(True)
        self.Patients_button.setEnabled(True)
        self.clearData.setEnabled(True)


    def acceptParams(self):
        self.main = Spectra_Anal()
        input_data = self.Diapason_choose.currentText()
        research_name = self.Research_name.toPlainText()
        if self.pathText.toPlainText() == '':
            os.chdir(os.curdir)
            path = glob.glob("input_dpt\\*.dpt")
        else:
            path = self.pathText.toPlainText()
        if self.checkBox.isChecked():
            normalization = 'y'
        else:
            normalization = 'n'
        self.Accept_Button.setEnabled(False)
        self.Diapason_choose.setEnabled(False)
        self.pathText.setEnabled(False)
        self.Research_name.setEnabled(False)
        self.checkBox.setEnabled(False)
        self.directory_dpt.setEnabled(False)
        self.main.read_files(input_data=input_data, normalization=normalization, path=path)
        self.main.cutting_spectra_and_finding_ratio()
        self.main.sorting_ratio_and_waves_by_names()
        self.main.calculate_ratio()
        self.main.calculate_and_sort_eigenvalues_and_vectors()
        self.main.calculate_t_and_p_matrix()
        self.main.write_eigenvalues_and_eigenvectors_in_files(research_name=research_name)
        self.Scores_2D.setEnabled(True)
        self.Scores_3D.setEnabled(True)
        self.Loadings_2D.setEnabled(True)
        self.Average_all.setEnabled(True)
        self.Patients_button.setEnabled(True)
        self.clearData.setEnabled(True)
        der1 = self.main.derivative_function(self.main.all_samples_for_deivative)
        der2 = self.main.derivative_function(der1)
        print(der1, der2)

        fig = plt.figure()
        ax = plt.axes()
        x = der1[0]
        # ax.plot(x, der1[2])
        ax.plot(x, der2[2])
        plt.show()
        # self.main.show_graphic_of_eigenvalues_and_pc(self)


    def rewriteData(self):
        self.main = Spectra_Anal()
        self.Scores_2D.setEnabled(False)
        self.Scores_3D.setEnabled(False)
        self.Loadings_2D.setEnabled(False)
        self.Average_all.setEnabled(False)
        self.Patients_button.setEnabled(False)
        self.clearData.setEnabled(False)
        self.Accept_Button.setEnabled(True)
        self.Diapason_choose.setEnabled(True)
        self.pathText.setEnabled(True)
        self.Research_name.setEnabled(True)
        self.checkBox.setEnabled(True)
        self.directory_dpt.setEnabled(True)


    def Home(self):
        self.SecondWindow.close()
        self.MainWindow.show()


    def scores2D(self):
        self.ColumnWindow = QtWidgets.QMainWindow()
        self.ui = Ui_ColumnWindow()
        self.ui.Signal(self.main, signal=1)
        self.ui.setupUi(self.ColumnWindow, self.SecondWindow)
        self.ColumnWindow.show()


    def loadings2D(self):
        self.ColumnWindow = QtWidgets.QMainWindow()
        self.ui = Ui_ColumnWindow()
        self.ui.Signal(self.main, signal=2)
        self.ui.setupUi(self.ColumnWindow, self.SecondWindow)
        self.ColumnWindow.show()


    def scores3D(self):
        self.ColumnWindow = QtWidgets.QMainWindow()
        self.ui = Ui_ColumnWindow()
        self.ui.Signal(self.main, signal=3)
        self.ui.setupUi(self.ColumnWindow, self.SecondWindow)
        self.ColumnWindow.show()


    def openAverage(self):
        self.AverageWindow = QtWidgets.QMainWindow()
        self.ui = Ui_AverageWindow()
        self.ui.Signal(self.main)
        self.ui.setupUi(self.AverageWindow, self.SecondWindow)
        self.AverageWindow.show()


    def openPatient(self):
        self.PatientWindow = QtWidgets.QMainWindow()
        self.ui = Ui_PatientWindow()
        self.ui.Signal(self.main)
        self.ui.setupUi(self.PatientWindow, self.SecondWindow)
        self.PatientWindow.show()


    def setupUi(self, SecondWindow, MainWindow):
        self.SecondWindow = SecondWindow
        self.MainWindow = MainWindow
        SecondWindow.setObjectName("SecondWindow")
        SecondWindow.resize(730, 504)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        SecondWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(SecondWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 100, 271, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setMouseTracking(True)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setEnabled(True)
        self.label_3.setGeometry(QtCore.QRect(30, 220, 271, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setMouseTracking(True)
        self.label_3.setTabletTracking(False)
        self.label_3.setAcceptDrops(False)
        self.label_3.setAutoFillBackground(False)
        self.label_3.setScaledContents(False)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.Accept_Button = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.acceptParams())
        self.Accept_Button.setGeometry(QtCore.QRect(90, 360, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Accept_Button.setFont(font)
        self.Accept_Button.setObjectName("Accept_Button")
        self.Research_name = QtWidgets.QTextEdit(self.centralwidget)
        self.Research_name.setGeometry(QtCore.QRect(30, 170, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Research_name.setFont(font)
        self.Research_name.setObjectName("Research_name")
        self.Error_Message = QtWidgets.QLabel(self.centralwidget)
        self.Error_Message.setGeometry(QtCore.QRect(20, 400, 271, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Error_Message.setFont(font)
        self.Error_Message.setText("")
        self.Error_Message.setObjectName("Error_Message")
        self.Diapason_choose = QtWidgets.QComboBox(self.centralwidget)
        self.Diapason_choose.setEnabled(True)
        self.Diapason_choose.setGeometry(QtCore.QRect(30, 290, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Diapason_choose.setFont(font)
        self.Diapason_choose.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.Diapason_choose.setAcceptDrops(False)
        self.Diapason_choose.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Diapason_choose.setEditable(True)
        self.Diapason_choose.setObjectName("Diapason_choose")
        self.Return_home = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.Home())
        self.Return_home.setGeometry(QtCore.QRect(430, 440, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Return_home.setFont(font)
        self.Return_home.setObjectName("Return_home")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setEnabled(True)
        self.label_4.setGeometry(QtCore.QRect(340, 10, 361, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setMouseTracking(True)
        self.label_4.setTabletTracking(False)
        self.label_4.setAcceptDrops(False)
        self.label_4.setAutoFillBackground(False)
        self.label_4.setScaledContents(False)
        self.label_4.setObjectName("label_4")
        self.Scores_2D = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.scores2D())
        self.Scores_2D.setEnabled(False)
        self.Scores_2D.setGeometry(QtCore.QRect(340, 90, 361, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Scores_2D.setFont(font)
        self.Scores_2D.setObjectName("Scores_2D")
        self.Loadings_2D = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.loadings2D())
        self.Loadings_2D.setEnabled(False)
        self.Loadings_2D.setGeometry(QtCore.QRect(340, 160, 361, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Loadings_2D.setFont(font)
        self.Loadings_2D.setObjectName("Loadings_2D")
        self.Scores_3D = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.scores3D())
        self.Scores_3D.setEnabled(False)
        self.Scores_3D.setGeometry(QtCore.QRect(340, 230, 361, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Scores_3D.setFont(font)
        self.Scores_3D.setObjectName("Scores_3D")
        self.Average_all = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.openAverage())
        self.Average_all.setEnabled(False)
        self.Average_all.setGeometry(QtCore.QRect(340, 300, 361, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Average_all.setFont(font)
        self.Average_all.setObjectName("Average_all")
        self.Patients_button = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.openPatient())
        self.Patients_button.setEnabled(False)
        self.Patients_button.setGeometry(QtCore.QRect(340, 370, 361, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Patients_button.setFont(font)
        self.Patients_button.setObjectName("Patients_button")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(300, 0, 20, 771))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.clearData = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.rewriteData())
        self.clearData.setEnabled(False)
        self.clearData.setGeometry(QtCore.QRect(60, 450, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.clearData.setFont(font)
        self.clearData.setObjectName("clearData")
        self.directory_dpt = QtWidgets.QToolButton(self.centralwidget)
        self.directory_dpt.setEnabled(True)
        self.directory_dpt.setGeometry(QtCore.QRect(250, 50, 31, 31))
        self.directory_dpt.setObjectName("directory_dpt")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 271, 41))
        self.label.setObjectName("label")
        self.pathText = QtWidgets.QTextEdit(self.centralwidget)
        self.pathText.setEnabled(True)
        self.pathText.setGeometry(QtCore.QRect(30, 50, 251, 31))
        self.pathText.setObjectName("pathText")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setEnabled(True)
        self.checkBox.setGeometry(QtCore.QRect(30, 330, 251, 21))
        self.checkBox.setChecked(True)
        self.checkBox.setTristate(False)
        self.checkBox.setObjectName("checkBox")
        self.label_2.raise_()
        self.label_3.raise_()
        self.Accept_Button.raise_()
        self.Research_name.raise_()
        self.Error_Message.raise_()
        self.Diapason_choose.raise_()
        self.Return_home.raise_()
        self.label_4.raise_()
        self.Scores_2D.raise_()
        self.Loadings_2D.raise_()
        self.Scores_3D.raise_()
        self.Average_all.raise_()
        self.Patients_button.raise_()
        self.line.raise_()
        self.clearData.raise_()
        self.label.raise_()
        self.pathText.raise_()
        self.directory_dpt.raise_()
        self.checkBox.raise_()
        SecondWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(SecondWindow)
        self.statusbar.setObjectName("statusbar")
        SecondWindow.setStatusBar(self.statusbar)

        self.retranslateUi(SecondWindow)
        QtCore.QMetaObject.connectSlotsByName(SecondWindow)

    def retranslateUi(self, SecondWindow):
        _translate = QtCore.QCoreApplication.translate
        SecondWindow.setWindowTitle(_translate("SecondWindow", "MainWindow"))
        self.label_2.setText(_translate("SecondWindow", "<html><head/><body><p>Введите название исследования:<br/>"
                                                        "(По умолчанию время запуска программы)</p></body></html>"))
        self.label_3.setText(_translate("SecondWindow", "<html><head/><body><p>Введите изучаемый диапазон волновых "
                                                        "чисел:<br/>(в формате &quot;xx-xxxx, yy-yyyy, ...&quot; )"
                                                        "</p></body></html>"))
        self.Accept_Button.setText(_translate("SecondWindow", "Accept"))
        self.Research_name.setHtml(_translate("SecondWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\""
                                                              " \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                              "<html><head><meta name=\"qrichtext\" content=\"1\" "
                                                              "/><style type=\"text/css\">\n"
                                                              "p, li { white-space: pre-wrap; }\n"
                                                              "</style></head><body style=\" font-family:\'Arial\',"
                                                              "\'Arial\'; font-size:12pt; font-weight:400;"
                                                              " font-style:normal;\">"
                                                              "\n""<p style=\"-qt-paragraph-type:empty; margin-top:0px;"
                                                              " margin-bottom:0px; margin-left:0px; margin-right:0px;"
                                                              " -qt-block-indent:0; text-indent:0px; font-family:"
                                                              "\'MS Shell Dlg 2\'; font-size:8.25pt;\"><br />"
                                                              "</p></body></html>"))
        self.Return_home.setText(_translate("SecondWindow", "Вернуться к началу"))
        self.label_4.setText(_translate("SecondWindow", "<html><head/><body><p align=\"center\">Расчёты произведены."
                                                        "<br/>Выберите необходимую опцию:</p></body></html>"))
        self.Scores_2D.setText(_translate("SecondWindow", "2D график по столбцам матрицы T (счета)"))
        self.Loadings_2D.setText(_translate("SecondWindow", "2D график по столбцам матрицы P (нагрузки)"))
        self.Scores_3D.setText(_translate("SecondWindow", "3D график по столбцам матрицы T (счета)"))
        self.Average_all.setText(_translate("SecondWindow", "Найти средние отношения поглощения"))
        self.Patients_button.setText(_translate("SecondWindow", "Найти отношения поглощений конкретных ММ"))
        self.clearData.setText(_translate("SecondWindow", "Сбросить параметры"))
        self.directory_dpt.setText(_translate("SecondWindow", "..."))
        self.label.setText(_translate("SecondWindow", "Текущий путь к .dpt спектрам:"))
        self.pathText.setHtml(_translate("SecondWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\""
                                                         " \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                         "<html><head><meta name=\"qrichtext\" content=\"1\" "
                                                         "/><style type=\"text/css\">\n""p,"
                                                         " li { white-space: pre-wrap; }\n"
                                                         "</style></head><body style=\" font-family:\'Arial\'; "
                                                         "font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                                         "<p style=\"-qt-paragraph-type:empty; margin-top:0px;"
                                                         " margin-bottom:0px; margin-left:0px; margin-right:0px;"
                                                         " -qt-block-indent:0; text-indent:0px;"
                                                         "\"><br /></p></body></html>"))
        self.checkBox.setText(_translate("SecondWindow", "Нормализация входных данных"))

