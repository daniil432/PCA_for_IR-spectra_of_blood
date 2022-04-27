from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QDialog


class AbWin(QDialog):
    def __init__(self):
        super(AbWin, self).__init__()
        loadUi("C:\\PCA_with_R\\AboutWindow.ui", self)
