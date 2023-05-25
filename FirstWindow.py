import PyQt5
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog
from SecondWindow import SecWin
from AboutWindow import AbWin
import sys
from PyQt5 import QtCore, QtWidgets


# if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
#     PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class FirstWin(QMainWindow):
    def __init__(self):
        super(FirstWin, self).__init__()
        loadUi("interface\\FirstWindow.ui", self)
        self.button_dpt.clicked.connect(self.openDpt)
        self.button_csv.setEnabled(False)
        self.directory_csv.setEnabled(False)
        self.directory_csv.clicked.connect(self.browseFiles)
        self.action_about_program.triggered.connect(self.aboutAss)
        self.show()

    def openDpt(self):
        ui_SecWin = SecWin(self)
        ui_SecWin.show()
        self.hide()

    def browseFiles(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open File', '.')
        self.path_csv = fname

    def aboutAss(self):
        ui_AbWin = AbWin()
        ui_AbWin.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ActiveWindow = FirstWin()
    sys.exit(app.exec())
