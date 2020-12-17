from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, Qt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys
from detect import detect
import os
from threeD_module import CthreeD
from preferences import Preferences
import qdarkstyle
import json
from faker import Faker
import csv
import joblib
from functools import partial

# Since I use macOS to develop the app and installed NuduleNet on Windows10 to run it,
# feel free to modify it for your own usage.
use_win10 = False
pkg_name = 'mac_pkg_copy.json'


if use_win10:

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import argparse
    from config import config

    sys.argv.append('eval')
    this_module = sys.modules[__name__]
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', '-m', metavar='NET', default=config['net'],
                        help='neural net')
    parser.add_argument("mode", type=str,
                        help="you want to test or val")
    parser.add_argument("--weight", type=str, default=config['initial_checkpoint'],
                        help="path to model weights to be used")


class NoduleCADx(QMainWindow):

    def __init__(self):
        super().__init__()
        loadUi('mainwindow.ui', self)
        self.setWindowFlags(Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)

        self.display_dialog = None
        # ColumnWidth of Patient List
        self.treeWidget.setColumnWidth(0, 70)
        self.treeWidget.setColumnWidth(1, 100)
        self.treeWidget.setColumnWidth(2, 100)
        self.treeWidget.setColumnWidth(3, 50)
        self.treeWidget.setColumnWidth(4, 50)
        # ColumnWidth of Scan and Nodule List
        self.noduletreeWidget.setColumnWidth(0, 70)
        self.noduletreeWidget.setColumnWidth(1, 100)
        self.noduletreeWidget.setColumnWidth(2, 100)
        self.noduletreeWidget.setColumnWidth(3, 50)
        self.noduletreeWidget.setColumnWidth(4, 100)
        self.noduletreeWidget.setColumnWidth(5, 100)
        self.noduletreeWidget.setColumnWidth(6, 100)
        self.noduletreeWidget.setColumnWidth(7, 100)
        self.noduletreeWidget.setColumnWidth(8, 100)
        self.preferences_dialog = None

        # pkg_name is the JSON file saved all the detected information (including scan path)
        # create a pkg_name.json if it doesn't exist.
        if not os.path.exists(pkg_name):
            with open(pkg_name, 'w') as json_file:
                initial_json = {'app': 'Nodule CADx', 'version': '1.0.0', "preferences":
                                {"threshold": "0.8",
                                 "project_directory": os.getcwd(),
                                 "automatic_classification": True,
                                 "windowlevel": "?"}, 'members': []}
                json.dump(initial_json, json_file, indent=2)

        # load pkg_name.json
        with open(pkg_name, 'r') as f:
            self.data = json.load(f)
        # load nodulenet and classification model and refresh patient list.
        self.nodulenet_model = None
        self.classification_model = None
        self.load_model()
        self.refresh_patient_list()

    def load_model(self):
        """
        Load (1)NoduleNet and (2)Classification model.
        Since I didn't install NoduleNet on my macOS, so only load it on Windows10.
        """
        # NoduleNet model
        if use_win10:
            args = parser.parse_args()
            initial_checkpoint = args.weight
            net = args.net
            net = getattr(this_module, net)(config)
            if initial_checkpoint:
                print('[Loading model from %s]' % initial_checkpoint)
                checkpoint = torch.load(initial_checkpoint, map_location='cpu')
                net.load_state_dict(checkpoint['state_dict'], )
            else:
                print('No model weight file specified')
            net.set_mode('eval')
            net.use_mask = True
            net.use_rcnn = True
            self.nodulenet_model = net
        else:
            self.nodulenet_model = None

        # Classification model
        self.classification_model = joblib.load('model/classification_model.pkl')

    @pyqtSlot()
    def on_reportButton_clicked(self):
        """
        Export system report in CSV format.
        """
        report = [['Name', 'Date of Birth', 'Sex', 'Final-Score', 'Management', 'Scan Path', 'Nodule', 'Diameter',
                   'Type', 'Calcification', 'Spiculation', 'Perifissural', 'Endobronchial', 'Score']]
        for m in self.data['members']:
            report.append([m['patient_name'], m['date_of_birth'], m['sex'], m['score'], m['management']])
            for s in m['scans']:
                report.append(['']*5 + [s['scan_path']])
                for i, n in enumerate(s['nodules'], start=1):
                    type_name = self.get_type_name(n['type'])
                    report.append(['']*6 + [f'Nodule{i}', n['diameter'], type_name, n['calcification'],
                                            n['spiculation'], n['perifissural'], n['endobronchial'], n['score']])
        with open('report.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(report)

    @staticmethod
    def get_type_name(nodule_type):
        """
        Get type name for display.
        """
        if nodule_type == '1':
            return 'non-solid'
        elif nodule_type == '2':
            return 'non/part'
        elif nodule_type == '3':
            return 'part-solid'
        elif nodule_type == '4':
            return 'part/solid'
        elif nodule_type == '5':
            return 'solid'

    def refresh_patient_list(self):
        """
        refresh patient list (upper block of main window).
        """
        self.treeWidget.clear()
        for m in self.data['members']:
            scan_item = QTreeWidgetItem(self.treeWidget,
                                        ['\u2713' if m['updated'] else '\u2717', m['patient_name'], m['date_of_birth'],
                                         m['sex'], m['score'], m['management']])
            for i in range(scan_item.columnCount()):
                scan_item.setTextAlignment(i, Qt.AlignHCenter)

    def refresh_scan_list(self, member):
        """
        refresh scan and nodule list (lower block of main window).
        """
        self.noduletreeWidget.clear()
        for scan in member['scans']:
            p = QTreeWidgetItem(self.noduletreeWidget, ['\u2713' if scan['updated'] else '\u2717', scan['scan_date'],
                                                        scan['scan_path']])
            if scan['updated']:
                for count, nodule in enumerate(scan['nodules'], start=1):
                    type_name = self.get_type_name(nodule['type'])
                    n_item = QTreeWidgetItem(p, ['', '', f'Nodule{count}', str(nodule['prob']), str(nodule['diameter']),
                                                 type_name, str(nodule['calcification']), str(nodule['spiculation']),
                                                 str(nodule['perifissural']), str(nodule['endobronchial']),
                                                 nodule['score']])
                    for i in range(n_item.columnCount()):
                        n_item.setTextAlignment(i, Qt.AlignHCenter)
            for i in range(p.columnCount()):
                p.setTextAlignment(i, Qt.AlignHCenter)
        self.noduletreeWidget.expandAll()

    @pyqtSlot()
    def on_actionPreferences_triggered(self):
        if not self.preferences_dialog:
            self.preferences_dialog = Preferences(self.data['preferences'])
            self.preferences_dialog.rejected.connect(self.update_preferences)
        self.preferences_dialog.show()

    def update_preferences(self):
        """
        update preferences when OK in preferences dialog is clicked.
        """
        self.data['preferences'] = self.preferences_dialog.preferences

    def on_treeWidget_itemClicked(self):
        index_member = self.treeWidget.currentIndex().row()
        self.refresh_scan_list(member=self.data['members'][index_member])

    @pyqtSlot()
    def on_loadscanButton_clicked(self):
        fname, _filter = QFileDialog.getOpenFileName(self, 'open file', '~/Desktop', 'Scan (*.mhd *.nrrd)')

        # make up some patient information for .mhd file from LUNA16
        faker = Faker()
        patient_name = faker.name()
        birth_date = faker.date()
        patient_sex = faker.profile()['sex']
        scan_date = faker.date()

        # For general DICOM series
        '''
        reader = sitk.ImageSeriesReader()
        reader = sitk.ImageSeriesReader()
        dir = '/Users/apple/Desktop/神農/一些參考/Patient CT EDU (Anonym-TR123)'
        seriesIDs = reader.GetGDCMSeriesIDs(dir)
        dcm_series = reader.GetGDCMSeriesFileNames(dir, seriesIDs[0])
        reader.SetFileNames(dcm_series)
        reader.MetaDataDictionaryArrayUpdateOn()
        # Execute() is needed to GetMetaData
        img = reader.Execute()
        patient_name = reader.GetMetaData(0,'0010|0010').strip()
        birth_date = reader.GetMetaData(0,'0010|0030').strip()
        patient_sex = reader.GetMetaData(0,'0010|0040').strip()
        '''

        exist = False
        for i, m in enumerate(self.data['members']):
            if patient_name == m['patient_name']:
                self.data['members'][i]['scans'].append({'updated': False, 'scan_path': fname,
                                                         'scan_date': scan_date, 'nodules': []})
                self.data['members'][i]['updated'] = False
                exist = True
        if not exist:
            self.data['members'].append(
                {'updated': False, 'patient_name': patient_name, 'date_of_birth': birth_date, 'sex': patient_sex,
                 'score': '?', 'management': '?', 'scans': [{'updated': False, 'scan_path': fname,
                                                             'scan_date': scan_date, 'nodules': []}]})
        self.refresh_patient_list()

    @pyqtSlot()
    def on_displayButton_clicked(self):
        index_member = self.treeWidget.currentIndex().row()
        nodule_select = None
        if self.noduletreeWidget.selectedItems()[0].parent():
            directory = self.noduletreeWidget.selectedItems()[0].parent().text(2)
            nodule_select = self.noduletreeWidget.currentIndex().row()
            index_scan = self.noduletreeWidget.indexFromItem(self.noduletreeWidget.selectedItems()[0].parent()).row()
        else:
            directory = self.noduletreeWidget.selectedItems()[0].text(2)
            index_scan = self.noduletreeWidget.indexFromItem(self.noduletreeWidget.selectedItems()[0]).row()

        self.display_dialog = CthreeD()
        self.display_dialog.updata_data_signal.connect(partial(self.update_data, index_member, index_scan))
        self.display_dialog.show()
        self.display_dialog.w = self.display_dialog.imgLabel_1.width()
        self.display_dialog.h = self.display_dialog.imgLabel_1.height()
        self.display_dialog.load_dicomfile(directory=directory, nodule_select=nodule_select,
                                           scan=self.data['members'][index_member]['scans'][index_scan])

    def update_data(self, index_member, index_scan, data_csv):
        self.data['members'][index_member]['scans'][index_scan]['nodules'] = []
        for row in data_csv:
            self.data['members'][index_member]['scans'][index_scan]['nodules'].append(row)
        self.refresh_scan_list(member=self.data['members'][index_member])

        self.management(index_member)

    def mousePressEvent(self, event):
        if app.focusWidget():
            self.setFocus()

    # TODO Not complete enough
    def management(self, index_member=None):
        """
        Get highest Lung-RADS score and match the date to show management
        """
        # diameter for solid component is needed for class 4 if nodule type is part-solid
        scores = []
        scans_date = []
        max_solid_component_diameter = 0
        for s in self.data['members'][index_member]['scans']:
            y, m, d = s['scan_date'].split(sep='-')
            scans_date.append(datetime(int(y), int(m), int(d)))
            for n in s['nodules']:
                scores.append(n['score'])
                if n['type'] == '3':
                    if eval(n['diameter']) * 0.5 > max_solid_component_diameter:
                        max_solid_component_diameter = eval(n['diameter']) * 0.5
        newest = datetime(1000, 1, 1)
        for scan_date in scans_date:
            if scan_date > newest:
                newest = scan_date
        management = ''
        if '?' in scores:
            max_score = '?'
            management = '?'
        else:
            breaker = False
            max_score = '0'
            for s in ['4X', '4B', '4A', '3', '2', '1', '0']:
                if scores.__len__() == 0:
                    print('no nodule')
                    max_score = '1'
                    break

                for score in scores:
                    if score == s:
                        max_score = s
                        breaker = True
                        break
                if breaker:
                    break

            if max_score == '0':
                management = 'Additional lung cancer screening CT images and/or comparison to ' \
                             'prior chest CT examinations is needed'
            elif max_score == '1' or max_score == '2':
                management = f'LDCT @ {newest.date()+relativedelta(years=1)}'
            elif max_score == '3':
                management = f'LDCT @ {newest.date()+relativedelta(months=6)}'
            elif max_score == '4A':
                management = f'LDCT @ {newest.date()+relativedelta(months=3)}'
                if max_solid_component_diameter >= 8:
                    management += ' (PET/CT may be used)'
            elif max_score == '4B' or max_score == '4X':
                management = f'Chest CT w/ or w/o contrast, PET/CT and/or tissue sampling may be used'
                # TODO 這邊是如果有新生大結節才要的
                # management += '1 month LDCT may be recommended to
                # address potentially infectious or inflammatory conditions'

        self.data['members'][index_member]['score'] = max_score
        self.data['members'][index_member]['management'] = management
        self.refresh_patient_list()

    @pyqtSlot()
    def on_detectButton_clicked(self):
        # show status on statusbar
        self.statusBar().showMessage('Model predicting, please wait for a while ...')
        self.statusBar().repaint()
        QApplication.instance().processEvents()

        # TODO Check if selected scan is already detected
        index_member = self.treeWidget.currentIndex().row()
        index_scan = self.noduletreeWidget.currentIndex().row()

        if use_win10:
            csv_data = detect(self.data['members'][index_member]['scans'][index_scan]['scan_path'],
                              self.nodulenet_model, self.classification_model, self.data['preferences'])
        else:
            csv_data = None

        self.update_data(index_member, index_scan, csv_data)
        self.data['members'][index_member]['scans'][index_scan]['updated'] = True
        self.refresh_scan_list(self.data['members'][index_member])
        status = [scan['updated'] for scan in self.data['members'][index_member]['scans']]
        if False not in status:
            self.data['members'][index_member]['updated'] = True
            self.refresh_patient_list()
        self.statusBar().showMessage('Done.', msecs=5000)
        self.management(index_member)

    @pyqtSlot()
    def on_savechangesButton_clicked(self):
        messagebox = QMessageBox.warning(self, 'Are you sure you want to save changes?',
                                         'You cannot undo this action, re-detect scans if necessary.',
                                         QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if messagebox == QMessageBox.Ok:
            with open(pkg_name, 'w') as json_file:
                json.dump(self.data, json_file, indent=2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    window = NoduleCADx()
    window.show()
    sys.exit(app.exec_())
