import sys
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QApplication,
    QLabel,
    QMessageBox,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QDialog,
)
from PyQt5 import QtCore, QtGui
import pandas as pd
import shutil
import os

import torch
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from PIL import Image 

class cats_vs_dogs(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(cats_vs_dogs, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1) # Initialize VGG16 model with pre-trained weights
vgg.eval() # disables features like dropout and batch normalization updates

pre_trained = cats_vs_dogs(1000, hidden_size=512, output_size=1, dropout=0.3)
pre_trained.load_state_dict(torch.load('final_model'))

# Convert resized images to tensors and normalize pixel values
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class PredictTab(QWidget):
    def __init__(self, parent):
        super(PredictTab, self).__init__(parent)
        self.setFixedSize(600, 500)
        self.imgPath = []
        self.imgIndex = 0
        self.predictions = []
        mainLayout = QVBoxLayout(self)

        self.imgLabel = QLabel()
        self.imgLabel.setStyleSheet(
            "background-color: lightgrey; border: 1px solid gray;"
        )
        self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.prevButton = QPushButton("<")
        self.prevButton.setMaximumWidth(50)
        self.prevButton.setEnabled(False)
        self.nextButton = QPushButton(">")
        self.nextButton.setMaximumWidth(50)
        self.nextButton.setEnabled(False)
        self.predLabel = QLabel("None")
        self.predLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.predLabel.setFixedWidth(300)
        self.predLabel.setFixedHeight(20)
        hWidget1 = QWidget(self)
        hWidget1.setFixedHeight(20)
        hLayout1 = QHBoxLayout(hWidget1)
        hLayout1.setContentsMargins(0, 0, 0, 0)
        hWidget2 = QWidget(self)
        hWidget2.setFixedHeight(25)
        hLayout2 = QHBoxLayout(hWidget2)
        hLayout2.setContentsMargins(0, 0, 0, 0)
        hWidget3 = QWidget(self)
        hWidget3.setFixedHeight(25)
        hLayout3 = QHBoxLayout(hWidget3)
        hLayout3.setContentsMargins(0, 0, 0, 0)
        hWidget4 = QWidget(self)
        hWidget4.setFixedHeight(25)
        hLayout4 = QHBoxLayout(hWidget4)
        hLayout4.setContentsMargins(0, 0, 0, 0)
        # hWidget.setStyleSheet("border: 1px solid red; padding: 0 0 0 0; margin: 0px;")

        loadButton = QPushButton("Select picture(s)")
        predButton = QPushButton("Predict")
        exportButton = QPushButton("Export")
        loadButton.clicked.connect(self.loadImg)
        self.prevButton.clicked.connect(self.prevImg)
        self.nextButton.clicked.connect(self.nextImg)
        predButton.clicked.connect(self.predict)
        exportButton.clicked.connect(self.export)

        mainLayout.addWidget(self.imgLabel)
        hLayout1.addWidget(self.prevButton)
        hLayout1.addWidget(self.predLabel)
        hLayout1.addWidget(self.nextButton)
        hLayout2.addWidget(loadButton)
        hLayout3.addWidget(predButton)
        hLayout3.addWidget(exportButton)
        mainLayout.addWidget(hWidget1)
        mainLayout.addWidget(hWidget2)
        mainLayout.addWidget(hWidget3)
        mainLayout.addWidget(hWidget4)

    def loadImg(self):
        dialog = QFileDialog()
        dialog.setWindowTitle("Select an image")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_():
            self.imgPath = [str(i) for i in dialog.selectedFiles()]
            self.predictions = [None for i in range(len(self.imgPath))]
            self.imgIndex = 0
            print("Selection:")
            for i in self.imgPath:
                print(i)
            self.updatePixmap(self.imgPath[self.imgIndex])
            self.prevButton.setEnabled(False)
            if len(self.imgPath) > 1:
                self.nextButton.setEnabled(True)
            elif len(self.imgPath) == 1:
                self.nextButton.setEnabled(False)
            self.updatePixmap(self.imgPath[self.imgIndex])
            # if self.cnn is not None:
            # self.predict()

    def updatePixmap(self, path, pred=1000):
        self.imgLabel.setPixmap(QtGui.QPixmap(path).scaled(500, 500))
        # self.imgLabel.setScaledContents(True)
        self.predLabel.setText(str(self.predictions[self.imgIndex]))
        if pred < 0.5:
            self.predLabel.setText(
                f"I think it's a Cat! Confidence: {(1.0-pred)*100:.0f}%"
            )
        elif pred > 0.5 and pred != 1000:
            self.predLabel.setText(
                f"I think it's a Dog! Confidence: {pred*100:.0f}%"
            )
        else:
            self.predLabel.setText("I don't know yet ")

    def predict(self):
        if len(self.imgPath) > 0:
            img = vgg(torch.unsqueeze(data_transforms(Image.open(self.imgPath[self.imgIndex])), 0))
            try:
                self.predictions[self.imgIndex] = pre_trained(img).detach().numpy()[0][0]
                self.updatePixmap(
                    self.imgPath[self.imgIndex], self.predictions[self.imgIndex]
                )
            except:
                QMessageBox(
                    QMessageBox.Warning,
                    "Error",
                    "Cannot convert image, please select a valid image",
                ).exec_()
        else:
            QMessageBox(
                QMessageBox.Warning,
                "Error",
                "Please select an image and a neural network model before making prediction",
            ).exec_()

    def nextImg(self):
        self.imgIndex += 1
        self.updatePixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == len(self.imgPath) - 1:
            self.nextButton.setEnabled(False)
        else:
            self.nextButton.setEnabled(True)
        self.prevButton.setEnabled(True)

    def prevImg(self):
        self.imgIndex -= 1
        self.updatePixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == 0:
            self.prevButton.setEnabled(False)
        else:
            self.prevButton.setEnabled(True)
        self.nextButton.setEnabled(True)

    def export(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save as", "Export.csv", "All (.csv)", options=options
        )

        if fname:
            data_tuples = list(zip(self.imgPath, self.predictions))
            df = pd.DataFrame(data_tuples, columns=["Images", "Predictions"])
            df.to_csv(fname)
            print(fname, "saved")

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setWindowTitle("Dog & Cat classifier")
        self.setFixedSize(600, 500)
        centralwidget = QWidget(self)
        PredictTab(centralwidget)
        self.setCentralWidget(centralwidget)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
