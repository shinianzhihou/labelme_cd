import PIL.Image
import PIL.ImageEnhance
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from .. import utils


class BrightnessContrastDialog(QtWidgets.QDialog):
    def __init__(self, img, callback, parent=None, img_1=None):
        super(BrightnessContrastDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Brightness/Contrast")

        self.slider_brightness = self._create_slider()
        self.slider_contrast = self._create_slider()
        self.slider_brightness_1 = self._create_slider()
        self.slider_contrast_1 = self._create_slider()

        formLayout = QtWidgets.QFormLayout()
        formLayout.addRow(self.tr("Brightness"), self.slider_brightness)
        formLayout.addRow(self.tr("Contrast"), self.slider_contrast)
        formLayout.addRow(self.tr("Brightness_1"), self.slider_brightness_1)
        formLayout.addRow(self.tr("Contrast_1"), self.slider_contrast_1)
        self.setLayout(formLayout)

        assert isinstance(img, PIL.Image.Image)
        assert isinstance(img_1, PIL.Image.Image)
        self.img = img
        self.img_1 = img_1
        
        self.callback = callback

    def onNewValue(self, value):
        brightness = self.slider_brightness.value() / 50.0
        contrast = self.slider_contrast.value() / 50.0

        brightness_1 = self.slider_brightness_1.value() / 50.0
        contrast_1 = self.slider_contrast_1.value() / 50.0

        img = self.img
        img = PIL.ImageEnhance.Brightness(img).enhance(brightness)
        img = PIL.ImageEnhance.Contrast(img).enhance(contrast)

        img_1 = self.img_1
        img_1 = PIL.ImageEnhance.Brightness(img_1).enhance(brightness_1)
        img_1 = PIL.ImageEnhance.Contrast(img_1).enhance(contrast_1)

        img_data = utils.img_pil_to_data(img)
        img_data_1 = utils.img_pil_to_data(img_1)
        qimage = QtGui.QImage.fromData(img_data)
        qimage_1 = QtGui.QImage.fromData(img_data_1)
        self.callback(qimage, qimage_1)

    def _create_slider(self):
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(0, 150)
        slider.setValue(50)
        slider.valueChanged.connect(self.onNewValue)
        return slider
