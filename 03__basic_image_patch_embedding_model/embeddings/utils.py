from PIL import Image
import numpy as np

def load_image(path, size=None):
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize(size)
    return np.array(img)
