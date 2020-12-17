from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import numpy as np


class QPaintLabel3(QLabel):
    mpsignal = pyqtSignal(str)
    def __init__(self, parent):
        super(QLabel, self).__init__(parent)

        self.setMinimumSize(1, 1)
        self.setMouseTracking(False)
        self.image = None
        self.processedImage = None
        self.imgr, self.imgc = None, None
        self.imgpos_x, self.imgpos_y = None, None
        self.pos_x = 20
        self.pos_y = 20
        self.imgr, self.imgc = None, None
        # 遇到list就停，圖上的顯示白色只是幌子
        self.pos_xy = []
        # 十字的中心點！每個QLabel指定不同中心點，這樣可以用一樣的paintevent function
        self.crosscenter = [0, 0]
        self.mouseclicked = None
        self.sliceclick = False
        # 決定用哪種paintEvent的type, general代表一般的
        self.type = 'general'
        self.slice_loc = [0, 0, 0]
        self.zoom = False
        self.zoom_press = False
        self.zoom_corner = [[0,0]]

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        if event.buttons() == Qt.LeftButton:
            self.mousePressEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        self.crosscenter[0] = event.x()
        self.crosscenter[1] = event.y()
        self.mpsignal.emit(self.type)
        if self.zoom:
            self.zoom_press = True
            self.display_image(1)

    def display_image(self, window=1):
        w, h = self.width(), self.height()
        qformat = QImage.Format_Indexed8
        if len(self.processedImage.shape) == 3:  # rows[0], cols[1], channels[2]
            if (self.processedImage.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.processedImage, self.processedImage.shape[1], self.processedImage.shape[0],
                     self.processedImage.strides[0], qformat)
        # ratio是指邊長放大倍率
        ratio = 2
        if self.zoom:
            zoom_count = self.zoom_corner.__len__() - 1
            r, c = img.size().height() / ratio ** zoom_count, img.size().width() / ratio ** zoom_count
            if self.zoom_press:
                midx, midy = self.crosscenter[0] / w * c, self.crosscenter[1] / h * r
                x = self.zoom_corner[-1][0] + int(midx - c / ratio / 2)
                y = self.zoom_corner[-1][1] + int(midy - r / ratio /  2)
                self.zoom_corner.append([x, y])
                img = img.copy(x, y, int(c/ratio), int(r/ratio))
                self.zoom_press = False
            else:
                img = img.copy(self.zoom_corner[-1][0], self.zoom_corner[-1][1], int(c), int(r))
        if window == 1:
            self.setScaledContents(True)
            backlash = self.lineWidth() * 2
            self.setPixmap(QPixmap.fromImage(img).scaled(w - backlash, h - backlash, Qt.IgnoreAspectRatio))
            self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        # 利用一個QFont來設定drawText的格式
        loc = QFont()
        loc.setPixelSize(10)
        loc.setBold(True)
        loc.setItalic(True)
        loc.setPointSize(15)
        if self.pixmap() and self.type != 'full':
            painter = QPainter(self)
            pixmap = self.pixmap()
            painter.drawPixmap(self.rect(), pixmap)

            painter.setPen(QPen(Qt.magenta, 10))
            painter.setFont(loc)
            painter.drawText(5, self.height() - 5, 'x = %3d  ,  y = %3d  ,  z = %3d'
                             % (self.slice_loc[0]+1, self.slice_loc[1]+1, self.slice_loc[2]+1))

            if self.type == 'axial':
                # 畫直條
                painter.setPen(QPen(Qt.red, 3))
                painter.drawLine(self.crosscenter[0], 0, self.crosscenter[0], self.height())
                # 畫橫條
                painter.setPen(QPen(Qt.cyan, 3))
                painter.drawLine(0, self.crosscenter[1], self.width(), self.crosscenter[1])
                # 畫中心
                painter.setPen(QPen(Qt.yellow, 3))
                painter.drawPoint(self.crosscenter[0], self.crosscenter[1])

            elif self.type == 'sagittal':
                # 畫直條
                painter.setPen(QPen(Qt.cyan, 3))
                painter.drawLine(self.crosscenter[0], 0, self.crosscenter[0], self.height())
                # 畫橫條
                painter.setPen(QPen(Qt.yellow, 3))
                painter.drawLine(0, self.crosscenter[1], self.width(), self.crosscenter[1])
                # 畫中心
                painter.setPen(QPen(Qt.red, 3))
                painter.drawPoint(self.crosscenter[0], self.crosscenter[1])

            elif self.type == 'coronal':
                # 畫直條
                painter.setPen(QPen(Qt.red, 3))
                painter.drawLine(self.crosscenter[0], 0, self.crosscenter[0], self.height())
                # 畫橫條
                painter.setPen(QPen(Qt.yellow, 3))
                painter.drawLine(0, self.crosscenter[1], self.width(), self.crosscenter[1])
                # 畫中心
                painter.setPen(QPen(Qt.cyan, 3))
                painter.drawPoint(self.crosscenter[0], self.crosscenter[1])
