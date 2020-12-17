from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import os
from PyQt5.QtCore import pyqtSlot


class Preferences(QDialog):

    def __init__(self, preferences):
        super().__init__()
        loadUi('Preferences.ui', self)
        self.preferences = preferences
        self.set_view()
        self.thresholdLabel.setToolTip('Thresholding probability of detected nodules, \nlarger threshold leads to larger sensitivity and smaller false positive per scan.')

    def update_preferences(self):
        self.preferences['project_directory'] = os.getcwd()
        self.preferences['threshold'] = str(self.thresholdSpinBox.value())
        self.preferences['automatic_classification'] = self.automaticcheckBox.isChecked()

    def set_view(self):
        self.directoryEdit.setText(self.preferences['project_directory'])
        self.thresholdSpinBox.setValue(float(self.preferences['threshold']))
        self.automaticcheckBox.setChecked(self.preferences['automatic_classification'])

    def mousePressEvent(self, event):
        # if app.focusWidget():
        self.setFocus()

    @pyqtSlot()
    def on_directoryButton_clicked(self):
        self.directoryEdit.setText(QFileDialog.getExistingDirectory(self, 'choose output directory'))

    @pyqtSlot()
    def on_okButton_clicked(self):
        self.update_preferences()
        self.reject()

    @pyqtSlot()
    def on_cancelButton_clicked(self):
        self.set_view()
        self.reject()
