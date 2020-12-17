import sys
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import *
import os
import cv2
import numpy as np
import qdarkstyle
from skimage import measure
from pyqtgraph.opengl import MeshData, GLAxisItem
from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem
from functools import partial
import SimpleITK as sitk
# From export image only
from matplotlib import pyplot as plt
from matplotlib import animation


def HU2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype('uint8')

    return image_new


class CthreeD(QDialog):

    updata_data_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        loadUi('threeD_module.ui', self)
        self.processedvoxel = None
        self.v1, self.v2, self.v3 = None, None, None
        self.axial_hSlider.valueChanged.connect(self.updateimg)
        self.axial_vSlider.valueChanged.connect(self.updateimg)
        self.coronal_vSlider.valueChanged.connect(self.updateimg)
        self.display_mask = self.maskcheckBox.isChecked()
        self.imgLabel_1.type = 'axial'
        self.imgLabel_2.type = 'sagittal'
        self.imgLabel_3.type = 'coronal'
        self.fullimgLabel.type = 'full'

        self.axialGrid.setSpacing(0)
        self.saggitalGrid.setSpacing(0)
        self.coronalGrid.setSpacing(0)
        self.savesliceBox.setItemDelegate(QStyledItemDelegate(self.savesliceBox))

        # self.colormap_hBox.insertStretch(2)
        # self.colormap_hBox.insertSpacerItem(0, QSpacerItem(30, 0, QSizePolicy.Fixed,  QSizePolicy.Fixed))

        self.savesliceButton.clicked.connect(self.saveslice_clicked)
        self.imgLabel_1.mpsignal.connect(self.cross_center_mouse)
        self.imgLabel_2.mpsignal.connect(self.cross_center_mouse)
        self.imgLabel_3.mpsignal.connect(self.cross_center_mouse)

        self.cross_recalc = True
        self.w, self.h = 0, 0
        self.oldwidget = None

        self.GLViewWidget.addItem(GLAxisItem(size=QVector3D(4, 4, 4), glOptions='opaque'))
        self.item = None
        self.delete_index = []
        self.fullimage = None

        self.axialButton.setStyleSheet('QPushButton {min-width: 10px;  min-height: 10px;}')
        self.sagittalButton.setStyleSheet('QPushButton {min-width: 10px;  min-height: 10px;}')
        self.coronalButton.setStyleSheet('QPushButton {min-width: 10px;  min-height: 10px;}')

        self.axial_mask, self.sagittal_mask, self.coronal_mask, self.seg_mask = None, None, None, None
        self.scan, self.dir = None, None
        self.crop_boxes, self.mask_probs, self.detections = None, None, None

    # 用來存gif的...
    def save_gif(self):
        slow_interval = 40
        fig = plt.figure()
        ims = []
        plt.axis('off')
        for z in range(100, self.coronal_vSlider.maximum()-10):
            a_loc = z
            axial = cv2.cvtColor((self.processedvoxel[a_loc, :, :]).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            axial_mask = np.zeros_like(axial)
            axial_mask[:, :, 0][self.axial_mask[a_loc, :, :]] = 255
            axial = cv2.addWeighted(axial, 1, axial_mask, 1, 0)
            axial_mask2 = np.zeros_like(axial)
            axial_mask2[:, :, 0][self.seg_mask[a_loc, :, :]] = 255
            axial_mask2[:, :, 1][self.seg_mask[a_loc, :, :]] = 255
            axial = cv2.addWeighted(axial, 1, axial_mask2, 0.2, 0)
            im = plt.imshow(axial, animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=slow_interval, blit=True,
                                        repeat_delay=1000)
        ani.save(f'nodulenet_demo.gif', writer='pillow', dpi=1200)
        plt.close(fig)

    def cross_center_mouse(self, _type):
        self.cross_recalc = False
        if _type == 'axial':
            self.axial_hSlider.valueChanged.disconnect(self.updateimg)

            self.axial_hSlider.setValue(self.imgLabel_1.crosscenter[0] *
                                        self.axial_hSlider.maximum() / self.imgLabel_1.width())
            self.axial_vSlider.setValue(self.imgLabel_1.crosscenter[1] *
                                        self.axial_vSlider.maximum() / self.imgLabel_1.height())

            self.axial_hSlider.valueChanged.connect(self.updateimg)

        elif _type == 'sagittal':
            self.axial_vSlider.valueChanged.disconnect(self.updateimg)

            self.sagittal_hSlider.setValue(self.imgLabel_2.crosscenter[0] *
                                           self.sagittal_hSlider.maximum() / self.imgLabel_2.width())
            self.sagittal_vSlider.setValue(self.sagittal_vSlider.maximum() - self.imgLabel_2.crosscenter[1] *
                                           self.sagittal_vSlider.maximum() / self.imgLabel_2.height())

            self.axial_vSlider.valueChanged.connect(self.updateimg)

        elif _type == 'coronal':
            self.axial_hSlider.valueChanged.disconnect(self.updateimg)

            self.coronal_hSlider.setValue(self.imgLabel_3.crosscenter[0] *
                                          self.coronal_hSlider.maximum() / self.imgLabel_3.width())
            self.coronal_vSlider.setValue(self.coronal_vSlider.maximum() - self.imgLabel_3.crosscenter[1] *
                                          self.coronal_vSlider.maximum() / self.imgLabel_3.height())

            self.axial_hSlider.valueChanged.connect(self.updateimg)

        self.imgLabel_1.crosscenter = [
            self.axial_hSlider.value() * self.imgLabel_1.width() / self.axial_hSlider.maximum(),
            self.axial_vSlider.value() * self.imgLabel_1.height() / self.axial_vSlider.maximum()]
        self.imgLabel_2.crosscenter = [
            self.sagittal_hSlider.value() * self.imgLabel_2.width() / self.sagittal_hSlider.maximum(),
            (self.sagittal_vSlider.maximum() - self.sagittal_vSlider.value()) *
            self.imgLabel_2.height() / self.sagittal_vSlider.maximum()]
        self.imgLabel_3.crosscenter = [
            self.coronal_hSlider.value() * self.imgLabel_3.width() / self.coronal_hSlider.maximum(),
            (self.sagittal_vSlider.maximum() - self.coronal_vSlider.value()) *
            self.imgLabel_3.height() / self.coronal_vSlider.maximum()]

        self.cross_recalc = True

    def saveslice_clicked(self):
        fname, _filter = QFileDialog.getSaveFileName(self, 'save file', '~/untitled', "Image Files (*.jpg)")
        if fname:
            if self.savesliceBox.currentText() == 'Axial':
                cv2.imwrite(fname, self.imgLabel_1.processedImage)
            elif self.savesliceBox.currentText() == 'Saggital':
                cv2.imwrite(fname, self.imgLabel_2.processedImage)
            elif self.savesliceBox.currentText() == 'Coronal':
                cv2.imwrite(fname, self.imgLabel_3.processedImage)
            else:
                print('No slice be chosen')
        else:
            print('Error')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.w = self.imgLabel_1.width()
        self.h = self.imgLabel_1.height()

        if self.processedvoxel is not None:
            self.updateimg()

    def dicom_clicked(self):
        dname = QFileDialog.getExistingDirectory(self, 'choose dicom directory')
        print(dname)
        self.load_dicomfile(dname)

    def load_dicomfile(self, directory, nodule_select=None, scan=None):

        self.scan = scan
        self.dir = directory
        head, tail = os.path.split(directory)
        self.processedvoxel = HU2uint8(sitk.GetArrayFromImage(sitk.ReadImage(directory)))

        # TODO result位址是在preferences內設定的

        self.crop_boxes = np.load(f'result/{tail}/crop_boxes.npy')
        self.mask_probs = np.load(f'result/{tail}/mask_probs.npy', allow_pickle=True)
        # 由於detections的資料都存在package內了，為求統一就不用npy檔的
        self.detections = self.scan['nodules']

        self.update_shape()
        self.refresh_mask()
        self.set_archive(nodule_select=nodule_select)

        self.imgLabel_1.setMouseTracking(True)
        self.imgLabel_2.setMouseTracking(True)
        self.imgLabel_3.setMouseTracking(True)

    def set_archive(self, nodule_select=None):
        self.v = QVBoxLayout()
        self.v.setAlignment(Qt.AlignTop)
        self.v.setSpacing(10)
        self.groupBox.setLayout(self.v)
        # 用n當代號因為detections內是一個一個nodule
        for i, (n, b, p) in enumerate(zip(self.detections, self.crop_boxes, self.mask_probs)):
            w = loadUi('archive_widget.ui')
            # coord
            # TODO 到底xyz還zyx... 要搞清楚
            w.tableWidget.setItem(0, 0, QTableWidgetItem(f"{n['z']}, {n['y']}, {n['x']}"))
            w.tableWidget.setItem(1, 0, QTableWidgetItem(n['prob']))
            # diameter
            w.tableWidget.setItem(2, 0, QTableWidgetItem(n['diameter']))
            # type
            infomation = ['non-solid', 'non/part', 'part-solid', 'part/solid', 'solid']
            combox = QComboBox(w.tableWidget)
            combox.addItems(infomation)
            combox.setCurrentIndex(int(n['type'])-1)
            # QDarkStyle的combobox有bug，要加這句delegate才可以解決，don't know why
            combox.setItemDelegate(QStyledItemDelegate(combox))
            w.tableWidget.setCellWidget(3, 0, combox)
            # score
            w.tableWidget.setItem(4, 0, QTableWidgetItem(n['score']))
            # special
            w.tableWidget.setItem(5, 0, QTableWidgetItem(n['calcification']))
            w.tableWidget.setItem(6, 0, QTableWidgetItem(n['spiculation']))
            w.tableWidget.setItem(7, 0, QTableWidgetItem(n['perifissural']))
            w.tableWidget.setItem(8, 0, QTableWidgetItem(n['endobronchial']))
            # thumbnail
            w.imgLabel.coord = [int(n['z']), int(n['y']), int(n['x'])]
            w.deleteButton.setIcon(QIcon('resources/trashcan.png'))
            w.deleteButton.setIconSize(QSize(30, 30))
            w.deleteButton.clicked.connect(partial(self.delete_nodule, w, i))

            self.v.addWidget(w)
            w.imgLabel.processedImage = self.processedvoxel[b[1]:b[4], b[2]:b[5], (b[3]+b[6])//2].copy()
            w.imgLabel.display_image(1)

            w.imgLabel.mpsignal.connect(partial(self.gotonodule, w))
            w.imgLabel.index = i

            if nodule_select == i:
                w.imgLabel.mousePressEvent(event=None)

    def delete_nodule(self, archive_widget, i):
        if archive_widget == self.oldwidget:
            self.remove3d()
        archive_widget.setVisible(False)
        self.v.removeWidget(archive_widget)
        self.refresh_mask(i)

    @pyqtSlot()
    def on_zoomButton_clicked(self):
        self.fullimgLabel.zoom = True

    @pyqtSlot()
    def on_returnButton_clicked(self):
        self.stackedWidget.setCurrentIndex(0)
        self.fullimage = None
        self.full_vSlider.valueChanged.disconnect()
        self.full_vSlider.setValue(self.full_vSlider.maximum() // 2)
        self.fullimgLabel.zoom = False
        self.fullimgLabel.zoom_corner = [[0, 0]]
        self.update_shape()
        self.updateimg()

    @pyqtSlot()
    def on_axialButton_clicked(self):
        self.stackedWidget.setCurrentIndex(1)
        self.fullimage = 'axial'
        self.full_vSlider.setMaximum(self.sagittal_vSlider.maximum())
        self.full_vSlider.setValue(self.sagittal_vSlider.value())
        self.full_vSlider.valueChanged.connect(self.sagittal_vSlider.setValue)
        self.updateimg()

    @pyqtSlot()
    def on_sagittalButton_clicked(self):
        self.stackedWidget.setCurrentIndex(1)
        self.fullimage = 'sagittal'
        self.full_vSlider.setMaximum(self.axial_hSlider.maximum())
        self.full_vSlider.setValue(self.full_vSlider.value())
        self.full_vSlider.valueChanged.connect(self.axial_hSlider.setValue)
        self.updateimg()

    @pyqtSlot()
    def on_coronalButton_clicked(self):
        self.stackedWidget.setCurrentIndex(1)
        self.fullimage = 'coronal'
        self.full_vSlider.setMaximum(self.axial_vSlider.maximum())
        self.full_vSlider.setValue(self.full_vSlider.value())
        self.full_vSlider.valueChanged.connect(self.axial_vSlider.setValue)
        self.updateimg()

    @pyqtSlot()
    def on_confirmButton_clicked(self):
        # TODO (1)從現有archive把資料存起來，之後要回傳給main修改scan內容的 (2)更新self.crop_boxes, self.mask_probs，離開此畫面時要重存npy
        # 這邊要做的事類似'detect' function
        csv = []
        for i in range(self.v.count()):

            w = self.v.itemAt(i).widget()
            z, y, x = [s.strip() for s in w.tableWidget.item(0, 0).text().split(',')]
            prob = w.tableWidget.item(1, 0).text()
            diameter = w.tableWidget.item(2, 0).text()
            nodule_type = str(w.tableWidget.cellWidget(3, 0).currentIndex() + 1)
            score = w.tableWidget.item(4, 0).text()
            calcification = w.tableWidget.item(5, 0).text()
            spiculation = w.tableWidget.item(6, 0).text()
            perifissural = w.tableWidget.item(7, 0).text()
            endobronchial = w.tableWidget.item(8, 0).text()

            csv.append({
                "x": x, "y": y, "z": z, "prob": prob,
                "diameter": diameter, "type": nodule_type, "score": score,
                "calcification": calcification, "spiculation": spiculation, "perifissural": perifissural,
                "endobronchial": endobronchial
            })

        self.updata_data_signal.emit(csv)

        self.reject()

    def refresh_mask(self, i=None):
        # 更新self.crop_boxes, self.mask_probs
        if i is not None:
            self.delete_index.append(i)

        self.axial_mask = np.zeros_like(self.processedvoxel, dtype=bool)
        self.sagittal_mask = np.zeros_like(self.processedvoxel, dtype=bool)
        self.coronal_mask = np.zeros_like(self.processedvoxel, dtype=bool)
        self.seg_mask = np.zeros_like(self.processedvoxel, dtype=bool)

        for j, (b, p) in enumerate(zip(self.crop_boxes, self.mask_probs)):
            if j in self.delete_index:
                continue
            self.axial_mask[b[1]:b[4], b[2]:b[5], b[3]:b[6]] = True
            self.sagittal_mask[b[1]:b[4], b[2]:b[5], b[3]:b[6]] = True
            self.coronal_mask[b[1]:b[4], b[2]:b[5], b[3]:b[6]] = True
            # line width
            lw = 2
            self.axial_mask[b[1]:b[4], b[2] + lw:b[5] - lw, b[3] + lw:b[6] - lw] = False
            self.sagittal_mask[b[1] + lw:b[4] - lw, b[2] + lw:b[5] - lw, b[3]:b[6]] = False
            self.coronal_mask[b[1] + lw:b[4] - lw, b[2]:b[5], b[3] + lw:b[6] - lw] = False

            self.seg_mask[b[1]:b[4], b[2]:b[5], b[3]:b[6]][p > 0] = True

        self.updateimg()

    def gotonodule(self, widget, coord, index):
        if self.oldwidget is not None:
            self.oldwidget.imgLabel.setStyleSheet('border: 1px solid SteelBlue;')
        widget.imgLabel.setStyleSheet('border: 1px solid red;')
        self.oldwidget = widget

        self.axial_hSlider.setValue(coord[0] / self.v3 * self.axial_hSlider.maximum())
        self.axial_vSlider.setValue(coord[1] / self.v2 * self.axial_vSlider.maximum())
        self.coronal_vSlider.setValue(coord[2] / self.v1 * self.coronal_vSlider.maximum())

        self.updateimg()
        self.display3d(index)

    def remove3d(self):
        if self.item in self.GLViewWidget.items:
            self.GLViewWidget.removeItem(self.item)

    def display3d(self, index):

        self.remove3d()
        b = self.crop_boxes[index]
        voxel = np.zeros((b[4]-b[1], b[5]-b[2], b[6]-b[3]))
        voxel[self.seg_mask[b[1]:b[4], b[2]:b[5], b[3]:b[6]]] = 1

        # level=0.5意思是1和0交界，內插在0.5的概念
        verts, faces, norm, val = measure.marching_cubes_lewiner(voxel, level=0.5, spacing=(1, 1, 1),
                                                                 gradient_direction='descent', step_size=1,
                                                                 allow_degenerate=True)

        mesh = MeshData(verts / 4, faces)  # scale down - because camera is at a fixed position
        mesh._vertexNormals = norm

        self.item = GLMeshItem(meshdata=mesh, color=[1, 1, 1, 1], shader="normalColor", drawEdges = False, smooth=True)

        t1 = (mesh._vertexes[:, 0].max() + mesh._vertexes[:, 0].min()) / 2
        t2 = (mesh._vertexes[:, 1].max() + mesh._vertexes[:, 1].min()) / 2
        t3 = (mesh._vertexes[:, 2].max() + mesh._vertexes[:, 2].min()) / 2

        self.item.translate(-t1, -t2, -t3, local=True)
        self.GLViewWidget.addItem(self.item)

    def on_maskcheckBox_toggled(self, _bool):
        self.display_mask = _bool
        self.updateimg()

    def update_shape(self):
        self.axial_hSlider.valueChanged.disconnect(self.updateimg)
        self.axial_vSlider.valueChanged.disconnect(self.updateimg)
        self.coronal_vSlider.valueChanged.disconnect(self.updateimg)

        self.v1, self.v2, self.v3 = self.processedvoxel.shape

        self.sagittal_vSlider.setMaximum(self.v1-1)
        self.coronal_vSlider.setMaximum(self.v1-1)
        self.sagittal_hSlider.setMaximum(self.v2-1)
        self.axial_vSlider.setMaximum(self.v2-1)
        self.coronal_hSlider.setMaximum(self.v3-1)
        self.axial_hSlider.setMaximum(self.v3-1)
        self.sagittal_vSlider.setValue(self.sagittal_vSlider.maximum()//2)
        self.coronal_vSlider.setValue(self.coronal_vSlider.maximum()//2)
        self.sagittal_hSlider.setValue(self.sagittal_hSlider.maximum()//2)
        self.axial_vSlider.setValue(self.axial_vSlider.maximum()//2)
        self.coronal_hSlider.setValue(self.coronal_hSlider.maximum()//2)
        self.axial_hSlider.setValue(self.axial_hSlider.maximum()//2)

        self.axial_hSlider.valueChanged.connect(self.updateimg)
        self.axial_vSlider.valueChanged.connect(self.updateimg)
        self.coronal_vSlider.valueChanged.connect(self.updateimg)

    def updateimg(self):
        a_loc = self.coronal_vSlider.value()
        c_loc = self.axial_vSlider.value()
        s_loc = self.axial_hSlider.value()

        axial = cv2.cvtColor((self.processedvoxel[a_loc, :, :]).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        sagittal = cv2.cvtColor((self.processedvoxel[:, :, s_loc]).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        coronal = cv2.cvtColor((self.processedvoxel[:, c_loc, :]).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if self.display_mask:

            axial_mask = np.zeros_like(axial)
            axial_mask[:, :, 0][self.axial_mask[a_loc, :, :]] = 255
            sagittal_mask = np.zeros_like(sagittal)
            sagittal_mask[:, :, 0][self.sagittal_mask[:, :, s_loc]] = 255
            coronal_mask = np.zeros_like(coronal)
            coronal_mask[:, :, 0][self.coronal_mask[:, c_loc, :]] = 255

            axial = cv2.addWeighted(axial, 1, axial_mask, 1, 0)
            sagittal = cv2.addWeighted(sagittal, 1, sagittal_mask, 1, 0)
            coronal = cv2.addWeighted(coronal, 1, coronal_mask, 1, 0)

            axial_mask2 = np.zeros_like(axial)
            axial_mask2[:, :, 0][self.seg_mask[a_loc, :, :]] = 255
            axial_mask2[:, :, 1][self.seg_mask[a_loc, :, :]] = 255
            sagittal_mask2 = np.zeros_like(sagittal)
            sagittal_mask2[:, :, 0][self.seg_mask[:, :, s_loc]] = 255
            sagittal_mask2[:, :, 1][self.seg_mask[:, :, s_loc]] = 255
            coronal_mask2 = np.zeros_like(coronal)
            coronal_mask2[:, :, 0][self.seg_mask[:, c_loc, :]] = 255
            coronal_mask2[:, :, 1][self.seg_mask[:, c_loc, :]] = 255

            axial = cv2.addWeighted(axial, 1, axial_mask2, 0.2, 0)
            sagittal = cv2.flip(cv2.addWeighted(sagittal, 1, sagittal_mask2, 0.2, 0), 0)
            coronal = cv2.flip(cv2.addWeighted(coronal, 1, coronal_mask2, 0.2, 0), 0)

        else:
            sagittal = cv2.flip(sagittal, 0)
            coronal = cv2.flip(coronal, 0)

        self.imgLabel_1.slice_loc = [s_loc, c_loc, a_loc]
        self.imgLabel_2.slice_loc = [s_loc, c_loc, a_loc]
        self.imgLabel_3.slice_loc = [s_loc, c_loc, a_loc]

        if self.cross_recalc:
            self.imgLabel_1.crosscenter = [self.w*s_loc//self.v3, self.h*c_loc//self.v2]
            self.imgLabel_2.crosscenter = [self.w*c_loc//self.v2,
                                           self.h*(self.sagittal_vSlider.maximum()-a_loc)//self.v1]
            self.imgLabel_3.crosscenter = [self.w*s_loc//self.v3,
                                           self.h*(self.sagittal_vSlider.maximum()-a_loc)//self.v1]

        if self.fullimage:

            self.fullimgLabel.processedImage = eval(self.fullimage)
            self.fullimgLabel.display_image(1)

        else:
            self.imgLabel_1.processedImage = axial
            self.imgLabel_2.processedImage = sagittal
            self.imgLabel_3.processedImage = coronal

            self.imgLabel_1.display_image(1)
            self.imgLabel_2.display_image(1)
            self.imgLabel_3.display_image(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    ex = CthreeD()
    ex.show()
    ex.w, ex.h = ex.imgLabel_1.width(), ex.imgLabel_1.height()
    sys.exit(app.exec_())
