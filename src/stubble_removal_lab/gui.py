"""Local desktop UI. Detection/inference run in a cancellable child process."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from uuid import uuid4

from PIL import Image
from PySide6.QtCore import QProcess,QStandardPaths,Qt,QTimer
from PySide6.QtWidgets import (
    QApplication,QButtonGroup,QCheckBox,QComboBox,QFileDialog,QFrame,QHBoxLayout,
    QLabel,QMainWindow,QMessageBox,QProgressBar,QPushButton,QSlider,QSplitter,
    QVBoxLayout,QWidget,
)

from .config import Profile
from .gui_canvas import PhotoCanvas
from .photos import open_photo
from .selection import make_profile,source_hash

STYLE="""
QMainWindow { background: #f5f7f8; }
QLabel { color: #253b43; font-size: 13px; }
QPushButton { background: white; border: 1px solid #ccd7db; border-radius: 7px; padding: 9px 14px; color: #253b43; }
QPushButton:hover { background: #edf7f5; border-color: #16897c; }
QPushButton:checked { background: #dff1ec; border-color: #16897c; color: #076b5e; }
QPushButton:disabled { color: #9daaad; background: #eef1f2; }
QPushButton#primary { background: #087f71; color: white; border: 0; font-weight: 600; padding: 13px; }
QPushButton#primary:disabled { background: #b6c7c4; }
QComboBox { background: white; border: 1px solid #ccd7db; padding: 8px; border-radius: 6px; }
QFrame#card { background: white; border: 1px solid #dce4e7; border-radius: 10px; }
QProgressBar { border: 0; background: #dce7e4; border-radius: 3px; min-height: 5px; max-height: 5px; }
QProgressBar::chunk { background: #16897c; }
"""


class Window(QMainWindow):
    def __init__(self,input_path=None,assets=None):
        super().__init__()
        self.setWindowTitle("祛胡青 · 本地照片工具")
        self.resize(1360,860)
        self.assets=assets
        self.source=None
        self.original=None
        self.full_result=None
        self.preview=None
        self.profile=None
        self.faces=[]
        self.job=None
        self.result_path=None
        self.pending_result=None
        self.worker_error=""
        self.stdout_buffer=""
        self.stderr_buffer=""
        self.cancelled=False
        self.callback=None
        self.cache=Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation))/"jobs"
        self.cache.mkdir(parents=True,exist_ok=True)
        self._build()
        self._controls()
        if input_path:
            QTimer.singleShot(0,lambda:self.open_path(Path(input_path)))

    def _build(self):
        central=QWidget()
        self.setCentralWidget(central)
        layout=QVBoxLayout(central)
        layout.setContentsMargins(22,18,22,18)
        header=QHBoxLayout()
        title=QLabel("祛胡青")
        title.setStyleSheet("font-size: 27px; font-weight: 700;")
        header.addWidget(title)
        header.addWidget(QLabel("  自动定位 · 手动校正 · 照片留在本机"))
        header.addStretch()
        self.open_button=QPushButton("打开照片")
        self.open_button.clicked.connect(self.choose_photo)
        header.addWidget(self.open_button)
        layout.addLayout(header)
        body=QHBoxLayout()
        layout.addLayout(body,1)
        controls=QFrame()
        controls.setObjectName("card")
        controls.setFixedWidth(264)
        side=QVBoxLayout(controls)
        side.setContentsMargins(16,20,16,18)
        side.setSpacing(12)
        side.addWidget(QLabel("1  定位需要处理的脸"))
        modes=QHBoxLayout()
        self.auto_button=QPushButton("自动定位")
        self.manual_button=QPushButton("手动选区")
        group=QButtonGroup(self)
        for button in (self.auto_button,self.manual_button):
            button.setCheckable(True)
            group.addButton(button)
            modes.addWidget(button)
        self.auto_button.setChecked(True)
        self.auto_button.clicked.connect(self.auto_detect)
        self.manual_button.clicked.connect(self.manual_select)
        side.addLayout(modes)
        self.face_combo=QComboBox()
        self.face_combo.currentIndexChanged.connect(self.select_face)
        side.addWidget(self.face_combo)
        self.repoint_button=QPushButton("重新标记三个点")
        self.repoint_button.clicked.connect(self.repoint)
        side.addWidget(self.repoint_button)
        self.inside_button=QPushButton("在框内自动定位")
        self.inside_button.clicked.connect(self.detect_inside)
        side.addWidget(self.inside_button)
        self.instruction=QLabel("打开一张照片开始。自动模式会定位人脸；手动模式框选后点三个点。")
        self.instruction.setWordWrap(True)
        self.instruction.setStyleSheet("color:#617780; line-height:1.5;")
        side.addWidget(self.instruction)
        side.addSpacing(12)
        side.addWidget(QLabel("2  生成与调整"))
        self.model_combo=QComboBox()
        self.model_combo.addItem("C · 自然修复（推荐）","C")
        self.model_combo.addItem("B · 对照版本","B")
        self.model_combo.currentIndexChanged.connect(self.invalidate_result)
        side.addWidget(self.model_combo)
        self.run_button=QPushButton("生成结果")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.process)
        side.addWidget(self.run_button)
        strength_row=QHBoxLayout()
        strength_row.addWidget(QLabel("修复强度"))
        strength_row.addStretch()
        self.strength_label=QLabel("100%")
        strength_row.addWidget(self.strength_label)
        side.addLayout(strength_row)
        self.strength=QSlider(Qt.Orientation.Horizontal)
        self.strength.setRange(0,100)
        self.strength.setValue(100)
        self.strength.valueChanged.connect(self.strength_changed)
        side.addWidget(self.strength)
        self.show_original=QCheckBox("右侧暂看原图")
        self.show_original.toggled.connect(self.render_result)
        side.addWidget(self.show_original)
        self.save_button=QPushButton("另存结果 PNG")
        self.save_button.clicked.connect(self.save_result)
        side.addWidget(self.save_button)
        side.addStretch()
        profile_row=QHBoxLayout()
        self.load_profile_button=QPushButton("加载定位")
        self.load_profile_button.clicked.connect(self.load_profile)
        self.save_profile_button=QPushButton("保存定位")
        self.save_profile_button.clicked.connect(self.save_profile)
        profile_row.addWidget(self.load_profile_button)
        profile_row.addWidget(self.save_profile_button)
        side.addLayout(profile_row)
        self.cancel_button=QPushButton("取消后台任务")
        self.cancel_button.clicked.connect(self.cancel)
        side.addWidget(self.cancel_button)
        body.addWidget(controls)
        photos=QVBoxLayout()
        nav=QHBoxLayout()
        self.photo_name=QLabel("尚未打开照片")
        nav.addWidget(self.photo_name,1)
        self.fit_button=QPushButton("看整张")
        self.fit_button.clicked.connect(self.fit_photos)
        self.focus_button=QPushButton("看脸部")
        self.focus_button.clicked.connect(self.focus_face)
        nav.addWidget(self.fit_button)
        nav.addWidget(self.focus_button)
        photos.addLayout(nav)
        split=QSplitter(Qt.Orientation.Horizontal)
        self.left=PhotoCanvas(editable=True)
        self.right=PhotoCanvas()
        for title,canvas in [("原图 / 定位",self.left),("修复结果",self.right)]:
            panel=QWidget()
            stack=QVBoxLayout(panel)
            stack.setContentsMargins(0,0,0,0)
            stack.addWidget(QLabel(title))
            stack.addWidget(canvas,1)
            split.addWidget(panel)
        split.setSizes([520,520])
        photos.addWidget(split,1)
        photos.addWidget(QLabel("滚轮缩放 · 拖动空白处平移 · 拖动彩色点微调定位"))
        body.addLayout(photos,1)
        self.left.instruction.connect(self.instruction.setText)
        self.left.boxSelected.connect(self.geometry_changed)
        self.left.pointsChanged.connect(self.geometry_changed)
        self.progress=QProgressBar()
        self.progress.setRange(0,0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status=QLabel("就绪")
        layout.addWidget(self.status)
        self.blend_timer=QTimer(self)
        self.blend_timer.setSingleShot(True)
        self.blend_timer.setInterval(100)
        self.blend_timer.timeout.connect(self.render_result)

    def _controls(self):
        busy=self.job is not None
        has_photo=self.source is not None
        for button in [self.open_button,self.auto_button,self.manual_button,self.load_profile_button,
                       self.model_combo,self.face_combo,self.repoint_button,self.inside_button]:
            button.setEnabled(not busy and (has_photo or button==self.open_button))
        self.left.setEnabled(not busy)
        self.run_button.setEnabled(not busy and len(self.left.points)==3 and self.left.box is not None)
        self.save_profile_button.setEnabled(not busy and len(self.left.points)==3)
        self.strength.setEnabled(not busy and self.full_result is not None)
        self.save_button.setEnabled(not busy and self.full_result is not None)
        self.show_original.setEnabled(self.full_result is not None)
        self.cancel_button.setEnabled(busy)
        self.progress.setVisible(busy)

    def choose_photo(self):
        filename,_=QFileDialog.getOpenFileName(self,"打开照片","","照片 (*.jpg *.jpeg *.JPG *.png *.PNG);;所有文件 (*)")
        if filename:
            self.open_path(Path(filename))

    def open_path(self,path):
        try:
            photo=open_photo(path)
        except Exception as error:
            return self.error(str(error))
        self.source=path.expanduser().resolve()
        self.original=photo
        self.profile=None
        self.faces=[]
        self.face_combo.clear()
        self.left.clear_photo()
        self.left.set_photo(photo)
        self.invalidate_result()
        self.photo_name.setText(f"{path.name}  ·  {photo.width} × {photo.height}")
        self._controls()
        if self.auto_button.isChecked():
            self.auto_detect()
        else:
            self.manual_select()

    def invalidate_result(self,*args):
        self.full_result=None
        self.preview=None
        self.result_path=None
        if hasattr(self,"right"):
            self.right.clear_photo()
            self._controls()

    def manual_select(self):
        if self.source is None:
            return
        self.manual_button.setChecked(True)
        self.profile=None
        self.invalidate_result()
        self.left.select_box()
        self._controls()

    def repoint(self):
        self.profile=None
        self.invalidate_result()
        self.left.select_points()
        self._controls()

    def geometry_changed(self,*args):
        self.profile=None
        self.invalidate_result()
        self._controls()

    def auto_detect(self):
        if self.source is None or self.job:
            return
        self.auto_button.setChecked(True)
        self.start_task({"task":"detect","source":str(self.source)},self.detection_done)

    def detect_inside(self):
        if self.left.box is None:
            return self.instruction.setText("请先在左图拖一个框圈住整张脸")
        self.start_task({"task":"detect","source":str(self.source),"box":self.left.box},self.detection_done)

    def detection_done(self,payload):
        self.faces=payload["faces"]
        self.face_combo.blockSignals(True)
        self.face_combo.clear()
        for index,face in enumerate(self.faces):
            suffix="" if face["points"] else " · 需手动标点"
            self.face_combo.addItem(f"人脸 {index+1}{suffix}")
        self.face_combo.blockSignals(False)
        self.face_combo.setCurrentIndex(0)
        self.select_face(0)

    def select_face(self,index):
        if not 0<=index<len(self.faces):
            return
        face=self.faces[index]
        self.invalidate_result()
        self.left.set_selection(face["box"],face["points"])
        self.profile=Profile.from_dict(face["profile"]) if face["profile"] else None
        self.focus_face()
        if self.profile:
            self.instruction.setText("定位完成。检查三个彩色点是否在嘴角和下巴；可拖动微调，然后生成结果。")
            self.status.setText(f"找到 {len(self.faces)} 张脸，当前选择第 {index+1} 张")
        else:
            self.left.select_points()
            self.status.setText(face.get("message","请手动标记三个点"))
        self._controls()

    def current_profile(self):
        if self.profile is not None:
            return self.profile
        if self.source is None or self.left.box is None:
            raise ValueError("请先打开照片并圈住脸部")
        return make_profile(self.source,self.left.box,self.left.points)

    def process(self):
        try:
            profile=self.current_profile()
        except Exception as error:
            return self.error(str(error))
        self.profile=profile
        directory=self.cache/uuid4().hex
        self.start_task({"task":"process","source":str(self.source),"profile":asdict(profile),
            "model":self.model_combo.currentData(),"assets":str(self.assets) if self.assets else None,
            "output":str(directory/"output")},self.process_done)

    def process_done(self,payload):
        self.result_path=Path(payload["path"])
        self.full_result=open_photo(self.result_path)
        self.show_original.setChecked(False)
        self.render_result()
        self.right.focus_box(self.left.box)
        self.status.setText("已生成。调整强度或暂看原图对照，满意后另存结果。")
        self._controls()

    def strength_changed(self,value):
        self.strength_label.setText(f"{value}%")
        self.blend_timer.start()

    def render_result(self,*args):
        if self.full_result is None or self.original is None:
            return
        self.preview=Image.blend(self.original,self.full_result,self.strength.value()/100)
        displayed=self.original if self.show_original.isChecked() else self.preview
        self.right.set_photo(displayed,fit=self.right.photo_item is None)

    def fit_photos(self):
        self.left.fit_photo()
        self.right.fit_photo()

    def focus_face(self):
        self.left.focus_box(self.left.box)
        self.right.focus_box(self.left.box)

    def save_result(self):
        if self.full_result is None:
            return
        self.render_result()
        filename,_=QFileDialog.getSaveFileName(self,"另存修复结果",str(self.source.with_name(self.source.stem+"_祛胡青.png")),"PNG照片 (*.png)")
        if not filename:
            return
        path=Path(filename)
        if path.suffix.lower()!=".png":
            path=path.with_suffix(".png")
        if path.resolve()==self.source:
            return self.error("请选择新的文件名，保留原图")
        try:
            options={"icc_profile":self.original.info["icc_profile"]} if self.original.info.get("icc_profile") else {}
            self.preview.save(path,**options)
            self.status.setText(f"已保存：{path}")
        except Exception as error:
            self.error(str(error))

    def load_profile(self):
        filename,_=QFileDialog.getOpenFileName(self,"加载定位配置","","定位配置 (*.json)")
        if not filename:
            return
        try:
            profile=Profile.load(Path(filename))
            if profile.source_sha256 and profile.source_sha256!=source_hash(self.source):
                raise ValueError("定位配置对应另一张照片，请使用当前照片的配置")
            if "crop_xyxy" in profile.parsing:
                box=profile.parsing["crop_xyxy"]
            else:
                a,b,c=profile.parsing["output_to_source_affine"][0]
                d,e,f=profile.parsing["output_to_source_affine"][1]
                corners=[(a*x+b*y+c,d*x+e*y+f) for x,y in [(0,0),(384,0),(0,384),(384,384)]]
                box=[min(x for x,y in corners),min(y for x,y in corners),max(x for x,y in corners),max(y for x,y in corners)]
            self.invalidate_result()
            self.left.set_selection(box,profile.points)
            self.profile=profile
            self.focus_face()
            self._controls()
            self.status.setText("已加载定位，可直接生成")
        except Exception as error:
            self.error(str(error))

    def save_profile(self):
        try:
            profile=self.current_profile()
            filename,_=QFileDialog.getSaveFileName(self,"保存定位",str(self.source.with_suffix(".stubble.json")),"定位配置 (*.json)")
            if filename:
                path=Path(filename)
                if path.resolve()==self.source:
                    raise ValueError("请选择新的配置文件名，保留原照片")
                path.write_text(json.dumps(asdict(profile),ensure_ascii=False,indent=2),encoding="utf-8")
                self.status.setText("已保存定位，下次可直接加载，也可供命令行使用")
        except Exception as error:
            self.error(str(error))

    def start_task(self,payload,callback):
        if self.job:
            return
        directory=self.cache/uuid4().hex
        directory.mkdir()
        request=directory/"request.json"
        request.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
        self.callback=callback
        self.pending_result=None
        self.worker_error=""
        self.stdout_buffer=""
        self.stderr_buffer=""
        self.cancelled=False
        self.job=QProcess(self)
        self.job.setProgram(sys.executable)
        self.job.setArguments(["-m","stubble_removal_lab.worker","--request",str(request)])
        self.job.readyReadStandardOutput.connect(self.read_stdout)
        self.job.readyReadStandardError.connect(self.read_stderr)
        self.job.finished.connect(self.task_finished)
        self.job.errorOccurred.connect(self.process_error)
        self._controls()
        self.status.setText("正在准备后台任务…")
        self.job.start()

    def read_stdout(self):
        if self.job is None:
            return
        self.stdout_buffer+=bytes(self.job.readAllStandardOutput()).decode("utf-8",errors="replace")
        while "\n" in self.stdout_buffer:
            line,self.stdout_buffer=self.stdout_buffer.split("\n",1)
            try:
                payload=json.loads(line)
            except ValueError:
                continue
            if payload.get("type")=="progress":
                self.status.setText(payload["message"])
            elif payload.get("type")=="result":
                self.pending_result=payload
            elif payload.get("type")=="error":
                self.worker_error=payload["message"]

    def read_stderr(self):
        if self.job:
            self.stderr_buffer=(self.stderr_buffer+bytes(self.job.readAllStandardError()).decode("utf-8",errors="replace"))[-12000:]

    def process_error(self,error):
        if error==QProcess.ProcessError.FailedToStart:
            self.worker_error="后台进程无法启动，请通过 uv run stubble gui 启动"
            self.task_finished(-1,QProcess.ExitStatus.CrashExit)

    def task_finished(self,code,status):
        if self.job is None:
            return
        self.read_stdout()
        self.read_stderr()
        job=self.job
        self.job=None
        job.deleteLater()
        self._controls()
        if self.cancelled:
            self.status.setText("任务已取消，可修改定位后重新生成")
        elif code==0 and self.pending_result:
            try:
                self.callback(self.pending_result)
            except Exception as error:
                self.error(str(error))
        else:
            message=self.worker_error or "后台任务未完成。请尝试手动选区；详细信息见终端。"
            if self.stderr_buffer:
                print(self.stderr_buffer,file=sys.stderr)
            self.error(message)

    def cancel(self):
        if self.job:
            self.cancelled=True
            self.job.kill()

    def error(self,message):
        self.status.setText(message)
        QMessageBox.warning(self,"需要调整",message)

    def closeEvent(self,event):
        if self.job:
            self.cancelled=True
            self.job.kill()
            self.job.waitForFinished(2000)
        event.accept()


def main(input_path=None,assets=None):
    app=QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("StubbleRemoval")
    app.setOrganizationName("iharee")
    app.setStyleSheet(STYLE)
    window=Window(input_path,assets)
    window.show()
    return app.exec()


if __name__=="__main__":
    raise SystemExit(main())
