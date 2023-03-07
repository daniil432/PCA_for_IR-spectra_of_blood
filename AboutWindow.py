from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QDialog


class AbWin(QDialog):
    def __init__(self):
        super(AbWin, self).__init__()
        loadUi("interface\\AboutWindow.ui", self)
