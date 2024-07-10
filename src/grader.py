import os
from PIL import Image
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

class CriticAgent:
    def __init__(self, directory):
        self.directory = directory

    def load_images(self):
        images = []
        filenames = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(self.directory, filename)
                img = Image.open(img_path)
                images.append(img)
                filenames.append(filename)
        return images, filenames

    def object_identification(self, images):
        # Dummy function for object identification (can use pre-trained models like YOLO, SSD, etc.)
        objects = ["object1", "object2"]
        return objects

    def color_identification(self, images, num_colors=3):
        colors = []
        for img in images:
            img = img.resize((100, 100))  # Resize for faster processing
            img_np = np.array(img)
            img_np = img_np.reshape((img_np.shape[0] * img_np.shape[1], 3))
            kmeans = KMeans(n_clusters=num_colors)
            kmeans.fit(img_np)
            centers = kmeans.cluster_centers_
            colors.append(centers)
        return colors

    def position_extraction(self, images):
        # Dummy function for position extraction
        positions = ["top-left", "bottom-right"]
        return positions

    def character_recognition(self, images):
        texts = []
        for img in images:
            text = pytesseract.image_to_string(img)
            texts.append(text)
        return texts