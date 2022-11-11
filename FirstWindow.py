from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog
from SecondWindow import SecWin
from AboutWindow import AbWin
import sys


class FirstWin(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("C:\\PCA_with_R\\interface\\FirstWindow.ui", self)
        self.button_dpt.clicked.connect(self.openDpt)
        self.button_csv.setEnabled(False)
        self.directory_csv.setEnabled(False)
        self.directory_csv.clicked.connect(self.browseFiles)
        self.action_about_program.triggered.connect(self.aboutAss)
        self.show()

    def openDpt(self):
        ui_SecWin = SecWin()
        ui_SecWin.exec()

    def browseFiles(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open File', 'C:\PCA_with_R')
        self.path_csv = fname

    def aboutAss(self):
        ui_AbWin = AbWin()
        ui_AbWin.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ActiveWindow = FirstWin()
    sys.exit(app.exec())
