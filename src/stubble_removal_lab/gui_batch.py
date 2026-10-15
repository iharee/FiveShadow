"""Desktop controls for still-image export and batch input selection."""
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import (
    QAbstractItemView,QCheckBox,QComboBox,QDialog,QDialogButtonBox,QFileDialog,
    QFormLayout,QHBoxLayout,QLabel,QLineEdit,QListWidget,QMessageBox,QPushButton,
    QSpinBox,QVBoxLayout,QWidget,
)

from .photos import FORMATS, PHOTO_EXTENSIONS


PHOTO_FILTER="照片 ("+" ".join("*"+suffix for suffix in sorted(PHOTO_EXTENSIONS))+");;所有文件 (*)"


class ExportControls(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        layout=QFormLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.format=QComboBox()
        self.format.addItem("保持输入格式","same")
        for name,encoding in FORMATS.items():
            self.format.addItem(encoding,name)
        self.quality=QSpinBox()
        self.quality.setRange(1,100)
        self.quality.setValue(95)
        self.format.setMinimumHeight(34)
        self.quality.setMinimumHeight(30)
        layout.addRow("保存格式",self.format)
        layout.addRow("JPEG/WebP质量",self.quality)
        self.format.setToolTip("PNG、TIFF和WebP保留透明度；JPEG/BMP使用白底，GIF最多256色。")
        self.quality.setToolTip("1到100；只影响JPEG/WebP。JPEG会重新压缩图片。")
        self.setMinimumHeight(layout.minimumSize().height())


class BatchDialog(QDialog):
    def __init__(self,parent=None,*,model="C",strength=100,output_format="same",quality=95):
        super().__init__(parent)
        self.setWindowTitle("批量处理照片")
        self.resize(680,560)
        layout=QVBoxLayout(self)
        description=QLabel("每张照片独立定位并处理一张脸。单张失败后继续，其余结果仍会保存。\n支持JPG、PNG、TIFF、WebP、BMP、静态GIF；暂不支持动画、多页或高位深图片。")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.inputs=QListWidget()
        self.inputs.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.inputs,1)
        row=QHBoxLayout()
        for text,callback in [("添加照片",self.add_files),("添加目录",self.add_directory),("移除选中",self.remove_selected)]:
            button=QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.recursive=QCheckBox("包含子目录")
        layout.addWidget(self.recursive)
        form=QFormLayout()
        self.model=QComboBox()
        for name in ("C","B","A","D"):
            self.model.addItem(name,name)
        self.model.setCurrentIndex(self.model.findData(model))
        self.strength=QSpinBox()
        self.strength.setRange(0,100)
        self.strength.setSuffix("%")
        self.strength.setValue(strength)
        self.face=QSpinBox()
        self.face.setRange(1,8)
        self.face.setToolTip("每张照片中按脸部大小排序，从1开始；不会跨照片识别同一个人。")
        form.addRow("修复模型",self.model)
        form.addRow("修复强度",self.strength)
        form.addRow("处理第几张脸",self.face)
        layout.addLayout(form)
        self.export=ExportControls()
        self.export.format.setCurrentIndex(self.export.format.findData(output_format))
        self.export.quality.setValue(quality)
        layout.addWidget(self.export)
        root=QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        self.output=QLineEdit(str(Path(root or Path.home())/"FiveShadow"/"batch"))
        self.output.setToolTip("结果目录，保留原照片；已存在时自动改用带_(1)编号的目录，不覆盖旧结果。每张照片各有独立的结果目录。")
        browse=QPushButton("选择保存位置")
        browse.clicked.connect(self.choose_output)
        row=QHBoxLayout()
        row.addWidget(self.output,1)
        row.addWidget(browse)
        layout.addLayout(row)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("开始批量处理")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_paths(self,paths):
        existing={self.inputs.item(index).text() for index in range(self.inputs.count())}
        for path in paths:
            value=str(Path(path).expanduser().resolve())
            if value not in existing:
                self.inputs.addItem(value)
                existing.add(value)

    def add_files(self):
        paths,_=QFileDialog.getOpenFileNames(self,"选择照片","",PHOTO_FILTER)
        self.add_paths(paths)

    def add_directory(self):
        path=QFileDialog.getExistingDirectory(self,"选择照片目录")
        if path:
            self.add_paths([path])

    def remove_selected(self):
        for item in self.inputs.selectedItems():
            self.inputs.takeItem(self.inputs.row(item))

    def choose_output(self):
        path=QFileDialog.getExistingDirectory(self,"选择结果目录的保存位置")
        if path:
            self.output.setText(str(Path(path)/Path(self.output.text()).name))

    def accept(self):
        if not self.inputs.count():
            QMessageBox.warning(self,"请选择照片","请添加照片或目录。")
            return
        if not self.output.text().strip():
            QMessageBox.warning(self,"请选择目录","请填写结果目录名称。")
            return
        super().accept()

    def request(self):
        return {"task":"batch", "inputs":[self.inputs.item(i).text() for i in range(self.inputs.count())],
                "output":self.output.text().strip(), "recursive":self.recursive.isChecked(),
                "model":self.model.currentData(), "strength":self.strength.value()/100,
                "face_index":self.face.value()-1, "output_format":self.export.format.currentData(),
                "quality":self.export.quality.value()}
