from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import numpy as np


class QThumbnail(QLabel):

    mpsignal = pyqtSignal(list, int)

    def __init__(self, parent):
        super(QLabel, self).__init__(parent)

        self.setMinimumSize(1, 1)
        self.setMouseTracking(False)
        self.processedImage = None
        self.imgr, self.imgc = None, None
        # 決定用哪種paintEvent的type, general代表一般的
        self.type = 'general'
        self.coord = []
        self.index = None

    def mousePressEvent(self, event: QMouseEvent):
        self.mpsignal.emit(self.coord, self.index)

    def display_image(self, window=1):
        self.imgr, self.imgc = self.processedImage.shape[0:2]
        qformat = QImage.Format_Indexed8

        if len(self.processedImage.shape) == 3:  # rows[0], cols[1], channels[2]
            if (self.processedImage.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.processedImage, self.processedImage.shape[1], self.processedImage.shape[0],
                     self.processedImage.strides[0], qformat)
        w, h = self.width(), self.height()
        if window == 1:
            self.setScaledContents(True)
            backlash = self.lineWidth() * 2
            self.setPixmap(QPixmap.fromImage(img).scaled(w - backlash, h - backlash, Qt.IgnoreAspectRatio))
            self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.update()
