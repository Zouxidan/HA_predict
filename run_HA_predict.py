import sys
from HA_predict import *


class mainwindow(QtWidgets.QMainWindow, Ui_HA_prediction):
    def __init__(self):
        super(mainwindow, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = mainwindow()
    main_window.show()
    sys.exit(app.exec())
