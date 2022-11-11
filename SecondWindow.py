import ColumnWindow
import AverageWindow
import PatientWindow
import AnalyzeSpectra
from PyQt5.uic import loadUi
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog


class SecWin(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("C:\\PCA_with_R\\interface\\SecondWindow.ui", self)
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
        self.pathText.setPlaceholderText('C:\PCA_with_R\input_dpt')
        self.Diapason_choose.setPlaceholderText("1700-1600")

    def browseFiles(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open File', 'C:\PCA_with_R')
        self.pathText.setText(fname)

    def acceptParams(self):
        if self.Diapason_choose.toPlainText() != '':
            input_range = self.Diapason_choose.toPlainText()
        else:
            input_range = self.Diapason_choose.placeholderText()
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

        research = AnalyzeSpectra.SpectraReader()
        self.filenames = research.read_spectra(path_dpt=path_dpt)
        matrix_w_r = research.cut_spectra(separate_df='y', input_range='1600-1650, 1580-1600, 1550-1580, 1520-1550, '
                                                                       '1500-1525, 1500-1510, 1440-1470')
        matrix_pca = research.cut_spectra(separate_df='n', input_range=input_range)

        average_res = AnalyzeSpectra.AverageAnal(self.filenames)
        self.ratio, self.waves = average_res.get_waves_ratios(matrix_w_r)
        ratio_waves = average_res.calc_average(self.ratio, self.waves)
        self.ratio_waves, self.ratio = average_res.normalize_average(ratio_waves, self.ratio)

        pca_res = AnalyzeSpectra.PcaAnal(matrix_pca)
        if normalization == 'y':
            pca_res.normalize()
        else:
            pass
        self.t_matrix, self.p_matrix = pca_res.performPCA()

        deriv1 = AnalyzeSpectra.derivative_df(matrix_pca)
        deriv2 = AnalyzeSpectra.derivative_df(deriv1)

        pca_deriv1 = AnalyzeSpectra.PcaAnal(deriv1)
        pca_deriv1.normalize()
        self.t_matrix_d1, self.p_matrix_d1 = pca_deriv1.performPCA()

        pca_deriv2 = AnalyzeSpectra.PcaAnal(deriv2)
        pca_deriv2.normalize()
        self.t_matrix_d2, self.p_matrix_d2 = pca_deriv2.performPCA()

        self.waves_loadings = [matrix_pca[matrix_pca.columns[0]], deriv1[deriv1.columns[0]], deriv2[deriv2.columns[0]]]

        self.Scores_2D.setEnabled(True)
        self.Scores_3D.setEnabled(True)
        self.Loadings_2D.setEnabled(True)
        self.Average_all.setEnabled(True)
        self.Patients_button.setEnabled(True)
        self.clearData.setEnabled(True)

    def scores2D(self):
        ui_ColWin = ColumnWindow.ColWin(signal=1, filenames=self.filenames, t_pca=self.t_matrix,
                                              p_pca=self.p_matrix, t_der1=self.t_matrix_d1,
                                              p_der1=self.p_matrix_d1, t_der2=self.t_matrix_d2,
                                              p_der2=self.p_matrix_d2, waves=self.waves_loadings)
        ui_ColWin.exec()

    def loadings2D(self):
        ui_ColWin = ColumnWindow.ColWin(signal=2, filenames=self.filenames, t_pca=self.t_matrix,
                                        p_pca=self.p_matrix, t_der1=self.t_matrix_d1,
                                        p_der1=self.p_matrix_d1, t_der2=self.t_matrix_d2,
                                        p_der2=self.p_matrix_d2, waves=self.waves_loadings)
        ui_ColWin.exec()

    def scores3D(self):
        ui_ColWin = ColumnWindow.ColWin(signal=3, filenames=self.filenames, t_pca=self.t_matrix,
                                        p_pca=self.p_matrix, t_der1=self.t_matrix_d1,
                                        p_der1=self.p_matrix_d1, t_der2=self.t_matrix_d2,
                                        p_der2=self.p_matrix_d2, waves=self.waves_loadings)
        ui_ColWin.exec()

    def openAverage(self):
        ui_AverWin = AverageWindow.AverWin(self.ratio_waves)
        ui_AverWin.exec()

    def openPatient(self):
        ui_PatWin = PatientWindow.PatWin(self.ratio, self.waves, self.ratio_waves, self.filenames)
        ui_PatWin.exec()
