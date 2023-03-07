import os.path
import pandas as pd
import ColumnWindow
import AverageWindow
import PatientWindow
import AnalyzeSpectra
from PyQt5.uic import loadUi
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog


class SecWin(QMainWindow):
    def __init__(self, parent):
        super(QMainWindow, self).__init__(parent)
        self.parent = parent
        self.tr_matrix = None
        self.pr_matrix = None
        self.waves_loadings = None
        self.p_matrix_d2 = None
        self.t_matrix_d2 = None
        self.p_matrix_d1 = None
        self.p_matrix = None
        self.t_matrix = None
        self.ratio = None
        self.ratio_waves = None
        self.waves = None
        self.filenames = None
        self.t_matrix_d1 = None
        loadUi("interface\\SecondWindow.ui", self)
        self.Accept_Button.setEnabled(True)
        self.Diapason_choose.setEnabled(True)
        self.pathText.setEnabled(True)
        self.Research_name.setEnabled(True)
        self.checkBox.setEnabled(True)
        self.directory_dpt.setEnabled(True)
        self.Scores_2D.setEnabled(False)
        self.Scores_3D.setEnabled(False)
        self.Loadings_2D.setEnabled(False)
        self.Average_all.setEnabled(False)
        self.Patients_button.setEnabled(False)
        self.Return_home.setEnabled(False)
        self.checkBox_save.setEnabled(False)
        self.pc_num.setEnabled(False)
        self.checkBox_gr.toggled.connect(self.checkBox_save.setEnabled)
        self.checkBox_gr.toggled.connect(self.pc_num.setEnabled)
        self.checkBox_gr.toggled.connect(lambda checked: not checked and self.checkBox_save.setChecked(False))

        self.Accept_Button.clicked.connect(self.acceptParams)
        self.Return_home.clicked.connect(self.closeEvent)
        self.Scores_2D.clicked.connect(self.scores2D)
        self.Loadings_2D.clicked.connect(self.loadings2D)
        self.Scores_3D.clicked.connect(self.scores3D)
        self.Average_all.clicked.connect(self.openAverage)
        self.Patients_button.clicked.connect(self.openPatient)
        self.directory_dpt.clicked.connect(self.browseFiles)
        self.Research_name.setPlaceholderText(str(datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
        self.pathText.setPlaceholderText(f'{os.path.abspath(os.curdir)}\\input_dpt')
        self.Diapason_choose.setPlaceholderText("1700-1600")

    def browseFiles(self):
        filename = QFileDialog.getExistingDirectory(self, 'Open File', '.')
        self.pathText.setText(filename)

    def acceptParams(self):
        if self.Diapason_choose.toPlainText() != '':
            input_range = self.Diapason_choose.toPlainText()
        else:
            input_range = self.Diapason_choose.placeholderText()
        research_name = self.Research_name.placeholderText()
        if self.pathText.toPlainText() != '':
            path_dpt = self.pathText.toPlainText()
        else:
            path_dpt = 'input_dpt'
        if self.checkBox.isChecked():
            normalization = True
        else:
            normalization = False
        pc_num = self.pc_num.value()

        self.Accept_Button.setEnabled(False)
        self.Diapason_choose.setEnabled(False)
        self.pathText.setEnabled(False)
        self.Research_name.setEnabled(False)
        self.checkBox.setEnabled(False)
        self.directory_dpt.setEnabled(False)
        self.checkBox_eigenvals.setEnabled(False)
        self.checkBox_gr.setEnabled(False)
        self.pc_num.setEnabled(False)
        self.checkBox_save.setEnabled(False)

        # Считываем спектры и создаём матрицы для обработки
        research = AnalyzeSpectra.SpectraReader()
        self.filenames = research.read_spectra(path_dpt=path_dpt)
        matrix_w_r = research.cut_spectra(separate_df=True, input_range='1600-1700, 1580-1620, 1550-1590, 1520-1550,'
                                          '1500-1525, 1497-1512, 1420-1480')
        matrix_pca = research.cut_spectra(separate_df=False, input_range=input_range)
        deriv1 = AnalyzeSpectra.derivative_df(matrix_pca)
        deriv2 = AnalyzeSpectra.derivative_df(deriv1)

        # Перевод данных на средние значения и подготовка их для дальнейшего анализа
        average_res = AnalyzeSpectra.AverageAnal(self.filenames)
        self.ratio, self.waves = average_res.get_waves_ratios(matrix_w_r)
        ratio_waves = average_res.calc_average(self.ratio, self.waves)
        self.ratio_waves, self.ratio = average_res.normalize_average(ratio_waves, self.ratio)
        ratio_norm = pd.DataFrame(self.ratio)
        ratio_norm = pd.concat([ratio_norm, pd.DataFrame(self.waves)], axis=1)

        # Создание объектов для обработки МГК и нормализация входных данных
        pca_res = AnalyzeSpectra.PcaAnal(matrix_pca)
        pca_deriv1 = AnalyzeSpectra.PcaAnal(deriv1)
        pca_deriv2 = AnalyzeSpectra.PcaAnal(deriv2)
        pca_ratio = AnalyzeSpectra.PcaAnal(ratio_norm, drop_first=False)
        if normalization == 'y':
            pca_res.normalize()
            pca_deriv1.normalize()
            pca_deriv2.normalize()
            pca_ratio.normalize()

        # Применение МГК и вывод необходимых данных
        #pca_res.graph_single(save=False)
        self.t_matrix, self.p_matrix = pca_res.performPCA()
        self.t_matrix_d1, self.p_matrix_d1 = pca_deriv1.performPCA()
        self.t_matrix_d2, self.p_matrix_d2 = pca_deriv2.performPCA()
        self.tr_matrix, self.pr_matrix = pca_ratio.performPCA()
        self.waves_loadings = [matrix_pca[matrix_pca.columns[0]], deriv1[deriv1.columns[0]], deriv2[deriv2.columns[0]]]

        if self.checkBox_save.isChecked():
            os.chdir('../projects')
            os.mkdir(research_name)
            os.chdir(research_name)

        # Отрисовка графиков собственных чисел
        if self.checkBox_eigenvals.isChecked():
            pca_res.eigen_graph(save=self.checkBox_save.isChecked(), name='origin')
            pca_deriv1.eigen_graph(save=self.checkBox_save.isChecked(), name='derivative_1')
            pca_deriv2.eigen_graph(save=self.checkBox_save.isChecked(), name='derivative_2')
            pca_ratio.eigen_graph(save=self.checkBox_save.isChecked(), name='ratio_pca')

        # Отрисовка тепловых карт и графиков ГК
        if self.checkBox_gr.isChecked():
            pca_res.heatmap_pca(self.t_matrix, pc_num, save=self.checkBox_save.isChecked(), name='origin')
            pca_deriv1.heatmap_pca(self.t_matrix_d1, pc_num, save=self.checkBox_save.isChecked(), name='derivative_1')
            pca_deriv2.heatmap_pca(self.t_matrix_d2, pc_num, save=self.checkBox_save.isChecked(), name='derivative_2')
            pca_ratio.heatmap_pca(self.tr_matrix, pc_num, save=self.checkBox_save.isChecked(), name='ratio_pca')

        self.Scores_2D.setEnabled(True)
        self.Scores_3D.setEnabled(True)
        self.Loadings_2D.setEnabled(True)
        self.Average_all.setEnabled(True)
        self.Patients_button.setEnabled(True)
        self.Return_home.setEnabled(True)

    def scores2D(self):
        ui_ColWin = ColumnWindow.ColWin(parent=self, signal=1, filenames=self.filenames,
                                        t_pca=self.t_matrix, p_pca=self.p_matrix,
                                        t_der1=self.t_matrix_d1, p_der1=self.p_matrix_d1,
                                        t_der2=self.t_matrix_d2, p_der2=self.p_matrix_d2,
                                        tr_pca=self.tr_matrix, pr_pca=self.pr_matrix,
                                        waves=self.waves_loadings)
        ui_ColWin.show()
        self.hide()

    def loadings2D(self):
        ui_ColWin = ColumnWindow.ColWin(parent=self, signal=2, filenames=self.filenames,
                                        t_pca=self.t_matrix, p_pca=self.p_matrix,
                                        t_der1=self.t_matrix_d1, p_der1=self.p_matrix_d1,
                                        t_der2=self.t_matrix_d2, p_der2=self.p_matrix_d2,
                                        tr_pca=self.tr_matrix, pr_pca=self.pr_matrix,
                                        waves=self.waves_loadings)
        ui_ColWin.show()
        self.hide()

    def scores3D(self):
        ui_ColWin = ColumnWindow.ColWin(parent=self, signal=3, filenames=self.filenames,
                                        t_pca=self.t_matrix, p_pca=self.p_matrix,
                                        t_der1=self.t_matrix_d1, p_der1=self.p_matrix_d1,
                                        t_der2=self.t_matrix_d2, p_der2=self.p_matrix_d2,
                                        tr_pca=self.tr_matrix, pr_pca=self.pr_matrix,
                                        waves=self.waves_loadings)
        ui_ColWin.show()
        self.hide()

    def openAverage(self):
        ui_AverWin = AverageWindow.AverWin(parent=self, ratio_waves=self.ratio_waves)
        ui_AverWin.show()
        self.hide()

    def openPatient(self):
        ui_PatWin = PatientWindow.PatWin(self, self.ratio, self.waves, self.ratio_waves, self.filenames)
        ui_PatWin.show()
        self.hide()

    def closeEvent(self, event):
        self.parent.show()
        self.close()
