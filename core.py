import io
import os
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from keras.models import Model
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

class Core:
    def __init__(self, model_filename: str = "model/drone-detection-v5.h5") -> None:
        self.model_path = model_filename
        self.labels_to_names = {0: 'drone'}
        self.model = models.load_model(self.model_path, backbone_name='resnet50')

    @staticmethod
    def load_image_by_path(filename: str) -> np.ndarray:
        return read_image_bgr(filename)

    @staticmethod
    def pre_process_image(image: np.ndarray) -> tuple:
        pre_processed_image = preprocess_image(image)
        resized_image, scale = resize_image(pre_processed_image)
        return resized_image, scale

    @staticmethod
    def predict(model: Model, image: np.ndarray, scale: float) -> tuple:
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        return boxes, scores, labels

    def draw_boxes_in_image(self, drawing_image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.3) -> list:
        detections = []
        for box, score in zip(boxes[0], scores[0]):
            if score < threshold:
                continue
            detections.append({"box": [int(coord) for coord in box], "score": int(score * 100)})
            color = label_color(0)
            b = box.astype(int)
            draw_box(drawing_image, b, color=color)
            caption = "{} {:.3f}".format(self.labels_to_names[0], score)
            draw_caption(drawing_image, b, caption)
        return detections
