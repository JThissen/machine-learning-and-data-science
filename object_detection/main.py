# %%
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
from typing import Any, List, Tuple
import matplotlib.pyplot as pyplot
import os
import sys
import numpy
import random

class CustomDataset(Dataset):
  def __init__(self, path: str) -> None:
    with open(path, "r") as file:
      self.all_lines = file.readlines()

    self.images_lines: List[str] = []
    self.labels_lines: List[str] = []

    for line in self.all_lines:
      if(sys.platform == "win32"):
        line = os.path.join("C:/", line[line.index("Users"):-1])

      self.images_lines.append(line.rstrip())
      line = line.replace("images", "labels").replace("jpg", "txt").replace("png", "txt")
      self.labels_lines.append(line.rstrip())
    
  def __len__(self) -> int:
    return len(self.images_lines)

  def __getitem__(self, index) -> (Image, numpy.ndarray):
    image = Image.open(self.images_lines[index]).convert("RGB")
    labels = numpy.loadtxt(self.labels_lines[index]) # label coords: [x_center, y_center, width, height]
    #TODO: add transforms
    return image, labels

def rescale_box(image: Image, label: numpy.ndarray) -> (Tuple[float, float, float, float]):
  label_x_center = label[1]
  label_y_center = label[2]
  label_width = label[3]
  label_height = label[4]
  image_width, image_height = image.size
  return (label_x_center * image_width, label_y_center * image_height, label_width * image_width, label_height * image_height)

def display_image(image: Image, label: numpy.ndarray, names: List[str]) -> None:
  imageDraw = ImageDraw.Draw(image)
  font = ImageFont.truetype("arial.ttf", 25)
  
  for line in label:
    color = numpy.random.randint(0, 255, 3)
    color_tuple = (color[0], color[1], color[2])
    
    x_center, y_center, width, height = rescale_box(image, line)
    x0 = x_center - (width / 2)
    y0 = y_center + (height / 2)
    x1 = x_center + (width / 2)
    y1 = y_center - (height / 2)
    imageDraw.rectangle([(x0, y0), (x1, y1)], outline=color_tuple, width=5)
    
    text = names[int(line[0])]
    text_size = font.getsize(text)
    text_offset = 10.0
    x0_text = x1 - text_size[0] - text_offset
    y0_text = y1
    imageDraw.text((x0_text, y0_text), text, fill=(255, 255, 255), font=font)
    
  pyplot.imshow(numpy.array(image))

def main() -> None:
  training_images_txt_path = os.path.join("./data/coco", "trainvalno5k.txt")
  dataset = CustomDataset(training_images_txt_path) # image (640, 480), label (8, 5)
  names_path = "./data/names.txt"
  names = List[str]
  with open(names_path, "r") as file:
    names = file.read().splitlines()
  image, label = dataset[0]
  display_image(image, label, names)

if __name__ == "__main__":
  main()
# %%
