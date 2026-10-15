"""Recovered semantic categories plus approximate CV protection masks."""
import cv2
import numpy as np
from PIL import Image, ImageDraw


def semantic_mask(probabilities, parsing_to_roi, template, width, height):
    classes = [cv2.warpAffine(np.clip(channel*255,0,255).astype(np.uint8),
                              parsing_to_roi[:2], (width,height), flags=cv2.INTER_LINEAR)
               for channel in probabilities]

    def combine(ids):
        combined = np.zeros((height,width), np.uint8)
        for index in ids:
            combined = cv2.bitwise_or(combined, classes[index])
        return cv2.blur(combined,(3,3))

    excluded = combine([1,11,15,0]) > 64
    mouth = combine([8,10]) > 32
    points = template * [width/768,height/512]
    nose = Image.new("L", (width,height))
    indices = list(range(61,66)) + list(range(68,73))[::-1]
    ImageDraw.Draw(nose).polygon([tuple(p) for p in points[indices]], fill=255)
    return (combine([2,12]).astype(np.float32)/255
            * (~excluded) * (~mouth) * (np.asarray(nose) == 0))
