"""Approximate three-point geometry, independent from the native 142-point fit."""
import cv2
import numpy as np
from PIL import Image

from .config import Profile

TEMPLATE_INDICES = [73, 83, 125]


def restoration_transform(points, template, width, height):
    control = np.column_stack([np.asarray(points, np.float64), np.ones(3)])
    if abs(np.linalg.det(control)) < 1e-6:
        raise ValueError("三个定位点不能共线")
    target = template[TEMPLATE_INDICES] * [width / 768, height / 512]
    forward = np.linalg.solve(control, target).T
    return np.vstack([forward, [0, 0, 1.]])


def align_image(photo, forward, width, height):
    inverse = np.linalg.inv(forward)
    return photo.transform((width, height), Image.Transform.AFFINE,
                           tuple(inverse[:2].ravel()), resample=Image.Resampling.BICUBIC)


def parsing_image(photo: Image.Image, profile: Profile):
    if "crop_xyxy" in profile.parsing:
        x0, y0, x1, y1 = profile.parsing["crop_xyxy"]
        if x0 < 0 or y0 < 0 or x1 > photo.width or y1 > photo.height:
            raise ValueError("分割裁剪区域超出照片范围")
        image = photo.crop((x0, y0, x1, y1)).resize((384, 384), Image.Resampling.BICUBIC)
        to_source = np.array([[(x1-x0)/384, 0, x0],
                              [0, (y1-y0)/384, y0], [0, 0, 1.]])
    else:
        to_source = np.vstack([profile.parsing["output_to_source_affine"], [0, 0, 1.]])
        if abs(np.linalg.det(to_source)) < 1e-10:
            raise ValueError("分割变换不可逆")
        image = photo.transform((384,384), Image.Transform.AFFINE,
                                tuple(to_source[:2].ravel()), resample=Image.Resampling.BICUBIC)
    return image, to_source


def add_correction(original, delta, source_to_roi, strength):
    """Add warped low-resolution RGB differences in 256-row memory bands."""
    height, width = original.shape[:2]
    inverse = np.linalg.inv(source_to_roi)
    output = np.empty_like(original)
    for y0 in range(0, height, 256):
        y1 = min(y0 + 256, height)
        transform = inverse[:2].copy()
        transform[1,2] -= y0
        correction = cv2.warpAffine(delta, transform, (width,y1-y0), flags=cv2.INTER_LINEAR)
        output[y0:y1] = np.clip(np.rint(
            original[y0:y1].astype(np.float32) + strength * correction
        ), 0, 255).astype(np.uint8)
    return output
