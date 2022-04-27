from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog


class TestWin(QDialog):
    def __init__(self):
        super(TestWin, self).__init__()
        loadUi("C:\\PCA_with_R\\Test.ui", self)
