# from albumentation docs

import cv2
import matplotlib.pyplot as plt

import torch

GT_COLOR = (0, 255, 0) # Green
PRED_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color, thickness=1):
    """Visualizes a single bounding box on the image"""
    bbox = [int(item) for item in bbox]
    x_min, y_min, x_max, y_max = bbox
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min), (x_min + text_width, y_min + int(text_height * 1.3)), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min+ int(text_height * 1.3)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name, color=PRED_COLOR, show=True):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, color)
    if show:
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    return img

def denorm(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std =  torch.tensor(std).view(-1, 1, 1)
    
    x = x * std + mean
    return x.permute(1, 2, 0)