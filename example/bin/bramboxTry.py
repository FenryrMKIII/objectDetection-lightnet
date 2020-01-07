import brambox
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


bbPath = Path(r"C:\Users\DAA426\myWork\objectDetection-lightnet\data\images\valves")
anno = []
images = []

for file in bbPath.iterdir():
    if file.suffix == ".txt":
        anno.append(file)
    elif file.suffix == ".png":
        images.append(file)

def getImageDims(id):
    root = Path(r"C:\Users\DAA426\myWork\objectDetection-lightnet\data\images\valves")
    im = Image.open(Path.joinpath(root, id + ".png"))
    width, height = im.size
    return (width, height)

class_label_map = ["valve"]

test = brambox.io.load(brambox.io.parser.annotation.DarknetParser(getImageDims, class_label_map), anno)

drawer = brambox.util.BoxDrawer(
    images=lambda img: Path.joinpath(bbPath, img + ".png"),  # Function to retrieve image path from image column name
    boxes=test,
    label=test.class_label,                 # Write class_label above boxes
)

for thing in drawer :
    plt.axis('off')
    plt.imshow(np.asarray(thing))
    plt.show()