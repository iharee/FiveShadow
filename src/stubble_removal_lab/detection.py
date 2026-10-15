"""YuNet proposals plus MediaPipe landmarks, mapped to oriented source pixels."""
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .detection_models import ensure_models
from .photos import open_photo
from .selection import make_profile, square_box


@dataclass
class FaceCandidate:
    box: list[int]
    points: list[list[float]] | None
    score: float | None
    profile: dict | None
    message: str = ""


def jaw_bottom_anchor(mesh):
    """Visual bottom of the lower jaw, matching the manual click definition.

    Landmark152 is the anatomical chin tip. With roll/yaw, that can be far
    sideways from the visually lowest jaw point used by the research affine.
    Interpolate the bottom of the projected jaw instead of assuming152.
    """
    contour=[172,136,150,149,176,148,152,377,400,378,379,365,397]
    jaw=mesh[contour]
    lowest=int(np.argmax(jaw[:,1]))
    anchor=jaw[lowest].copy()
    if 0<lowest<len(jaw)-1:
        neighbors=jaw[lowest-1:lowest+2]
        local_x=neighbors[:,0]-anchor[0]
        local_y=neighbors[:,1]-anchor[1]
        if np.min(np.abs(np.diff(np.sort(local_x))))>1e-3:
            a,b,c=np.polyfit(local_x,local_y,2)
            if a < -1e-8:
                dx=-b/(2*a)
                dy=a*dx*dx+b*dx+c
                if local_x.min()<=dx<=local_x.max() and 0<=dy<=max(2,float(np.ptp(neighbors[:,1]))):
                    anchor+=np.array([dx,dy])
    return anchor.tolist()


def _iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return intersection/max(1,area_a+area_b-intersection)


def detect_faces(source: Path, *, box=None, model_directory=None, progress=None):
    notify=progress or (lambda message:None)
    root=ensure_models(model_directory,notify)
    notify("正在定位人脸")
    photo=open_photo(source)
    rgb=np.asarray(photo)
    height,width=rgb.shape[:2]
    proposals=[]
    if box is not None:
        proposals=[(square_box(box,photo.size),None)]
    else:
        detector=cv2.FaceDetectorYN.create(str(root/"face_detection_yunet_2023mar.onnx"),
            "",(320,320),0.6,0.3,5000,cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU)
        # Two scales cover group/portrait photos without assuming face size.
        for limit in (1600,800):
            scale=min(1.0,limit/max(width,height))
            resized=cv2.resize(rgb,(max(32,round(width*scale)),max(32,round(height*scale))))
            detector.setInputSize((resized.shape[1],resized.shape[0]))
            _,faces=detector.detect(cv2.cvtColor(resized,cv2.COLOR_RGB2BGR))
            if faces is None:
                continue
            sx,sy=width/resized.shape[1],height/resized.shape[0]
            for row in faces:
                x,y,w,h=map(float,row[:4])
                rectangle=[max(0,x*sx),max(0,y*sy),min(width,(x+w)*sx),min(height,(y+h)*sy)]
                if rectangle[2]-rectangle[0]<10 or rectangle[3]-rectangle[1]<10:
                    continue
                if any(_iou(rectangle,other[0])>.4 for other in proposals):
                    continue
                proposals.append((rectangle,float(row[-1])))
        proposals.sort(key=lambda item:(item[0][2]-item[0][0])*(item[0][3]-item[0][1]),reverse=True)
        proposals=proposals[:8]
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker,FaceLandmarkerOptions,RunningMode

    options=FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(root/"face_landmarker.task")),
        running_mode=RunningMode.IMAGE,num_faces=1,min_face_detection_confidence=.4,
        min_face_presence_confidence=.4,output_face_blendshapes=False)
    candidates=[]
    with FaceLandmarker.create_from_options(options) as landmarker:
        # Close portraits may evade a proposal scale; full-frame mesh is a fallback.
        if not proposals:
            proposals=[([0,0,width,height],None)]
        for index,(proposal,score) in enumerate(proposals):
            notify(f"正在定位嘴角和下巴（{index+1}/{len(proposals)}）")
            landmarks=None
            bounds=None
            for padding in ((1.0,1.35) if box is not None else (1.8,2.3)):
                bounds=square_box(proposal,photo.size,padding=padding)
                x0,y0,x1,y1=bounds
                crop=photo.crop(tuple(bounds))
                crop.thumbnail((1024,1024))
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(np.asarray(crop)))
                result=landmarker.detect(image)
                if result.face_landmarks:
                    landmarks=result.face_landmarks[0]
                    break
            if landmarks is None:
                if score is not None or box is not None:
                    candidates.append(FaceCandidate(square_box(proposal,photo.size,padding=1.5),None,
                        score,None,"检测到了区域，但关键点未定位成功；请手动标记三个点"))
                continue
            x0,y0,x1,y1=bounds
            mesh=np.array([[x0+p.x*(x1-x0),y0+p.y*(y1-y0)] for p in landmarks],np.float64)
            if not np.isfinite(mesh).all() or len(mesh)<292:
                continue
            mesh[:,0]=np.clip(mesh[:,0],0,width-1)
            mesh[:,1]=np.clip(mesh[:,1],0,height-1)
            lips=sorted([mesh[61].tolist(),mesh[291].tolist()],key=lambda point:point[0])
            points=lips+[jaw_bottom_anchor(mesh)]
            mesh_box=[float(mesh[:,0].min()),float(mesh[:,1].min()),
                      float(mesh[:,0].max()),float(mesh[:,1].max())]
            face_box=square_box(mesh_box,photo.size,padding=1.6)
            profile=make_profile(source,face_box,points,origin="automatic",
                detection={"detector":"YuNet 2023mar + MediaPipe FaceLandmarker v1",
                           "mouth_landmark_indices":[61,291],
                           "chin_policy":"projected_jaw_bottom_v1",
                           "anatomical_chin":mesh[152].tolist(),
                           "selected_chin":points[-1],"proposal_score":score})
            candidates.append(FaceCandidate(face_box,points,score,asdict(profile)))
    if not candidates:
        raise ValueError("没有定位到可用人脸。请选择手动模式，框住脸部并标记左右嘴角和下巴")
    return candidates
