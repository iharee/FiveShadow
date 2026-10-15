"""Photo canvas with source-pixel box/point editing independent of zoom."""
import math

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView


class PhotoCanvas(QGraphicsView):
    boxSelected=Signal(object)
    pointsChanged=Signal(object)
    instruction=Signal(str)

    def __init__(self,editable=False,parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#e9eef0"))
        self.setRenderHints(QPainter.RenderHint.Antialiasing|QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.editable=editable
        self.photo_item=None
        self.box=None
        self.points=[]
        self.mode="view"
        self.anchor=None
        self.dragging_point=None
        self.overlays=[]
        self.auto_fit=True

    def set_photo(self,photo:Image.Image,fit=True):
        rgb=photo.convert("RGB")
        image=QImage(rgb.tobytes(),rgb.width,rgb.height,rgb.width*3,QImage.Format.Format_RGB888).copy()
        pixmap=QPixmap.fromImage(image)
        if self.photo_item is None:
            self.photo_item=self.scene().addPixmap(pixmap)
        else:
            self.photo_item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(0,0,rgb.width,rgb.height))
        if fit:
            self.fit_photo()

    def clear_photo(self):
        self.scene().clear()
        self.photo_item=None
        self.overlays=[]
        self.box=None
        self.points=[]

    def fit_photo(self):
        if self.photo_item:
            self.auto_fit=True
            self.fitInView(self.sceneRect(),Qt.AspectRatioMode.KeepAspectRatio)

    def focus_box(self,box):
        if self.photo_item and box:
            self.auto_fit=False
            rect=QRectF(box[0],box[1],box[2]-box[0],box[3]-box[1])
            pad=max(rect.width(),rect.height())*.12
            self.fitInView(rect.adjusted(-pad,-pad,pad,pad),Qt.AspectRatioMode.KeepAspectRatio)

    def set_selection(self,box,points):
        self.box=list(box) if box else None
        self.points=[list(p) for p in points] if points else []
        self.mode="adjust"
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.draw_selection()

    def select_box(self):
        if not self.photo_item or not self.editable:
            return
        self.mode="box"
        self.points=[]
        self.box=None
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.draw_selection()
        self.instruction.emit("拖一个框圈住整张脸：覆盖眉眼、鼻子、嘴唇到下巴")

    def select_points(self):
        if self.box is None:
            self.select_box()
            return
        self.points=[]
        self.mode="points"
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.draw_selection()
        self.pointsChanged.emit([])
        self.instruction.emit("第1/3步：点击画面左侧嘴角；可用滚轮放大")

    def draw_selection(self):
        for item in self.overlays:
            self.scene().removeItem(item)
        self.overlays=[]
        pen=QPen(QColor("#0f9b8e"),2)
        pen.setCosmetic(True)
        if self.box:
            x0,y0,x1,y1=self.box
            self.overlays.append(self.scene().addRect(QRectF(x0,y0,x1-x0,y1-y0),pen))
        colors=["#f6b73c","#40b6ef","#e67da0"]
        for index,(x,y) in enumerate(self.points):
            marker=self.scene().addEllipse(-6,-6,12,12,QPen(QColor("white"),1),QColor(colors[index]))
            marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            marker.setPos(x,y)
            marker.setZValue(10)
            marker.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            label=self.scene().addSimpleText(str(index+1))
            label.setBrush(QColor("white"))
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(x+3,y+3)
            label.setZValue(11)
            self.overlays.extend([marker,label])

    def _point(self,event):
        p=self.mapToScene(event.position().toPoint())
        rect=self.sceneRect()
        return [min(max(p.x(),0),max(0,rect.width()-1)),
                min(max(p.y(),0),max(0,rect.height()-1))]

    def mousePressEvent(self,event):
        if not self.editable or self.photo_item is None or event.button()!=Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        point=self._point(event)
        if self.mode=="box":
            self.anchor=point
            self.box=point+point
            self.draw_selection()
            event.accept()
            return
        if self.mode=="points" and len(self.points)<3:
            self.points.append(point)
            self.draw_selection()
            self.pointsChanged.emit([list(p) for p in self.points])
            if len(self.points)==1:
                self.instruction.emit("第2/3步：点击画面右侧嘴角")
            elif len(self.points)==2:
                self.instruction.emit("第3/3步：点击画面中下颌轮廓的最低处（侧脸时不是前突的下巴尖）")
            else:
                self.mode="adjust"
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.instruction.emit("定位完成。可拖动彩色点微调，然后点击「生成结果」")
            event.accept()
            return
        if self.mode=="adjust":
            position=event.position()
            for index,p in enumerate(self.points):
                screen=self.mapFromScene(QPointF(*p))
                if math.hypot(screen.x()-position.x(),screen.y()-position.y())<=14:
                    self.dragging_point=index
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self.anchor is not None:
            point=self._point(event)
            self.box=[min(self.anchor[0],point[0]),min(self.anchor[1],point[1]),
                      max(self.anchor[0],point[0]),max(self.anchor[1],point[1])]
            self.draw_selection()
            return
        if self.dragging_point is not None:
            self.points[self.dragging_point]=self._point(event)
            self.draw_selection()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self.anchor is not None:
            self.anchor=None
            if self.box[2]-self.box[0]<20 or self.box[3]-self.box[1]<20:
                self.box=None
                self.draw_selection()
                self.instruction.emit("框选区域太小，请重新拖动圈住整张脸")
                return
            self.box=[int(round(value)) for value in self.box]
            self.boxSelected.emit(list(self.box))
            self.select_points()
            return
        if self.dragging_point is not None:
            self.dragging_point=None
            self.pointsChanged.emit([list(p) for p in self.points])
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self,event):
        if self.photo_item:
            factor=1.2 if event.angleDelta().y()>0 else 1/1.2
            scale=self.transform().m11()*factor
            if .015<scale<24:
                self.auto_fit=False
                self.scale(factor,factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit_photo()
