from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
import matplotlib.pyplot as plt
import ColumnWindow
import AverageWindow
import PatientWindow


class SecWin(QDialog):
    def __init__(self, SpeAn):
        self.SpeAn = SpeAn
        super(SecWin, self).__init__()
        loadUi("C:\\PCA_with_R\\SecondWindow.ui", self)
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
        self.clearData.setEnabled(False)
        self.clearData.setEnabled(False)
        self.Return_home.setEnabled(False)
        self.Accept_Button.clicked.connect(self.acceptParams)
        # self.Return_home.clicked.connect(self.close)
        self.Scores_2D.clicked.connect(self.scores2D)
        self.Loadings_2D.clicked.connect(self.loadings2D)
        self.Scores_3D.clicked.connect(self.scores3D)
        self.Average_all.clicked.connect(self.openAverage)
        self.Patients_button.clicked.connect(self.openPatient)
        # self.clearData.clicked.connect(self.rewriteData)
        self.directory_dpt.clicked.connect(self.browseFiles)
        self.Research_name.setPlaceholderText(str(datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
        self.Diapason_choose.setPlaceholderText('1500-1525')
        self.pathText.setPlaceholderText('C:\PCA_with_R\input_dpt')

    def browseFiles(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open File', 'C:\PCA_with_R')
        self.pathText.setText(fname)

    def acceptParams(self):
        if self.pathText.toPlainText() != '':
            input_data = self.Diapason_choose.toPlainText()
        else:
            input_data = "1500-1525"
        research_name = self.Research_name.toPlainText()
        if self.pathText.toPlainText() != '':
            path_dpt = self.pathText.toPlainText()
        else:
            path_dpt = 'C:\PCA_with_R\input_dpt'
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
        self.Scores_2D.setEnabled(True)
        self.Scores_3D.setEnabled(True)
        self.Loadings_2D.setEnabled(True)
        self.Average_all.setEnabled(True)
        self.Patients_button.setEnabled(True)
        self.clearData.setEnabled(True)
        self.clearData.setEnabled(True)
        self.Return_home.setEnabled(True)
        self.SpeAn.read_files(input_data, normalization, path_dpt)
        self.SpeAn.cutting_spectra_and_finding_ratio()
        self.SpeAn.sorting_ratio_and_waves_by_names()
        self.SpeAn.calculate_ratio()
        self.SpeAn.calculate_and_sort_eigenvalues_and_vectors(self.SpeAn.input_matrix)
        self.t_matrix_pca, self.p_matrix_pca = self.SpeAn.calculate_t_and_p_matrix()
        self.SpeAn.show_graphic_of_eigenvalues_and_pc()

        der1graph = self.SpeAn.derivative_function(self.SpeAn.all_samples_for_derivative)
        der2graph = self.SpeAn.derivative_function(der1graph)

        xd, yd = der2graph[0], der2graph[1]
        plt.figure()
        plt.plot(xd, yd, '-', linewidth=4, label='Data')
        # plt.show()
        # self.SpeAn.fitting(der2graph, 4)
        der1pca = der1graph[1:]
        der2pca = der2graph[1:]
        self.waves = self.SpeAn.all_samples_for_derivative[0]
        self.SpeAn.derivative_saving(der2graph)
        self.SpeAn.calculate_and_sort_eigenvalues_and_vectors(der1pca)
        self.t_matrix_der1, self.p_matrix_der1 = self.SpeAn.calculate_t_and_p_matrix()
        self.SpeAn.calculate_and_sort_eigenvalues_and_vectors(der2pca)
        self.t_matrix_der2, self.p_matrix_der2 = self.SpeAn.calculate_t_and_p_matrix()
        self.waves_loadings = [self.SpeAn.one_wave, der1graph[0], der2graph[0]]
        self.SpeAn.write_eigenvalues_and_eigenvectors_in_files(research_name, self.t_matrix_pca, self.p_matrix_pca,
                                                              self.t_matrix_der1, self.p_matrix_der1,
                                                              self.t_matrix_der2, self.p_matrix_der2)
        self.Scores_2D.setEnabled(True)
        self.Scores_3D.setEnabled(True)
        self.Loadings_2D.setEnabled(True)
        self.Average_all.setEnabled(True)
        self.Patients_button.setEnabled(True)
        self.clearData.setEnabled(True)
        # self.SpeAn.show_graphic_of_eigenvalues_and_pc()

    def scores2D(self):
        ui_ColWin = ColumnWindow.ColWin(SpeAn=self.SpeAn, signal=1, t_pca=self.t_matrix_pca,
                                              p_pca=self.p_matrix_pca, t_der1=self.t_matrix_der1,
                                              p_der1=self.p_matrix_der1, t_der2=self.t_matrix_der2,
                                              p_der2=self.p_matrix_der2, waves=self.waves_loadings)
        ui_ColWin.exec()


    def loadings2D(self):
        ui_ColWin = ColumnWindow.ColWin(SpeAn=self.SpeAn, signal=2, t_pca=self.t_matrix_pca,
                                        p_pca=self.p_matrix_pca, t_der1=self.t_matrix_der1,
                                        p_der1=self.p_matrix_der1, t_der2=self.t_matrix_der2,
                                        p_der2=self.p_matrix_der2, waves=self.waves_loadings)
        ui_ColWin.exec()

    def scores3D(self):
        ui_ColWin = ColumnWindow.ColWin(SpeAn=self.SpeAn, signal=3, t_pca=self.t_matrix_pca,
                                        p_pca=self.p_matrix_pca, t_der1=self.t_matrix_der1,
                                        p_der1=self.p_matrix_der1, t_der2=self.t_matrix_der2,
                                        p_der2=self.p_matrix_der2, waves=self.waves_loadings)
        ui_ColWin.exec()

    def openAverage(self):
        ui_AverWin = AverageWindow.AverWin(SpeAn=self.SpeAn)
        ui_AverWin.exec()

    def openPatient(self):
        ui_PatWin = PatientWindow.PatWin(SpeAn=self.SpeAn)
        ui_PatWin.exec()
