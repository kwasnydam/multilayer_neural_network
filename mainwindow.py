import sys
from DataHolder import  DataHolder
from PyQt5 import QtGui, QtWidgets, QtCore, uic
form_class = uic.loadUiType("mainwindow.ui")[0]


class MyWindowClass(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        data = DataHolder(_filename= './IrisDataTrain.xls', _number_of_fetures=4)
        self._model = MyTableModel(self)
        self._model.update(data.rawdata.values)
        self.data_table.setModel(self._model)

    def btn_go_clicked(self):
        self.web.load(QtCore.QUrl(self.txt_url.text()))


class MyTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(MyTableModel, self).__init__()
        self.datatable = None

    def update(self, dataIn):
        print ('Updating Model')
        self.datatable = dataIn
        #print(str(dataIn[1,1]))
        print('Datatable : {0}'.format(self.datatable))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable[:, 1])

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable[1,:])

    def data(self, index, role=QtCore.Qt.DisplayRole):
        print('Data Call')
        print(index.column(), index.row())
        if role == QtCore.Qt.DisplayRole:
            i = index.row()
            j = index.column()
            # return QtCore.QVariant(str(self.datatable.iget_value(i, j)))
            x = str(self.datatable[i, j])
            print(x)
            return str(self.datatable[i, j])
        else:
            return QtCore.QVariant()

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled


app = QtWidgets.QApplication(sys.argv)
myWindow = MyWindowClass(None)
myWindow.show()
myWindow.raise_()
app.exec_()
