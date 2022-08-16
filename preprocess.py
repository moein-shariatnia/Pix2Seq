# Code from https://www.kaggle.com/code/billiemage/object-detection

import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from config import CFG

class XMLParser:
    def __init__(self,xml_file):

        self.xml_file = xml_file
        self._root = ET.parse(self.xml_file).getroot()
        self._objects = self._root.findall("object")
        # path to the image file as describe in the xml file
        self.img_path = os.path.join(CFG.img_path, self._root.find('filename').text)
        # image id 
        self.image_id = self._root.find("filename").text
        # names of the classes contained in the xml file
        self.names = self._get_names()
        # coordinates of the bounding boxes
        self.boxes = self._get_bndbox()

    def parse_xml(self):
        """"Parse the xml file returning the root."""
    
        tree = ET.parse(self.xml_file)
        return tree.getroot()

    def _get_names(self):

        names = []
        for obj in self._objects:
            name = obj.find("name")
            names.append(name.text)

        return np.array(names)

    def _get_bndbox(self):

        boxes = []
        for obj in self._objects:
            coordinates = []
            bndbox = obj.find("bndbox")
            coordinates.append(np.int32(bndbox.find("xmin").text))
            coordinates.append(np.int32(np.float32(bndbox.find("ymin").text)))
            coordinates.append(np.int32(bndbox.find("xmax").text))
            coordinates.append(np.int32(bndbox.find("ymax").text))
            boxes.append(coordinates)

        return np.array(boxes)

def xml_files_to_df(xml_files):
    
    """"Return pandas dataframe from list of XML files."""
    
    names = []
    boxes = []
    image_id = []
    xml_path = []
    img_path = []
    for f in xml_files:
        xml = XMLParser(f)
        names.extend(xml.names)
        boxes.extend(xml.boxes)
        image_id.extend([xml.image_id] * len(xml.names))
        xml_path.extend([xml.xml_file] * len(xml.names))
        img_path.extend([xml.img_path] * len(xml.names))
    a = {"image_id": image_id,
         "names": names,
         "boxes": boxes,
         "xml_path":xml_path,
         "img_path":img_path}
    
    df = pd.DataFrame.from_dict(a, orient='index')
    df = df.transpose()
    
    df['xmin'] = -1
    df['ymin'] = -1
    df['xmax'] = -1
    df['ymax'] = -1

    df[['xmin','ymin','xmax','ymax']] = np.stack([df['boxes'][i] for i in range(len(df['boxes']))])

    df.drop(columns=['boxes'], inplace=True)
    df['xmin'] = df['xmin'].astype('float32')
    df['ymin'] = df['ymin'].astype('float32')
    df['xmax'] = df['xmax'].astype('float32')
    df['ymax'] = df['ymax'].astype('float32')
    
    df['id'] = df['image_id'].map(lambda x: x.split(".jpg")[0])
    
    return df

def concat_gt(row):
    label = row['label']

    xmin = row['xmin']
    xmax = row['xmax']
    ymin = row['ymin']
    ymax = row['ymax']

    return [label, xmin, ymin, xmax, ymax]

def group_objects(df):
    df['concatenated'] = df.apply(concat_gt, axis=1)

    df = df.groupby('id')[['concatenated', 'img_path']].agg({'concatenated': list, 
                                                             'img_path': np.unique}).reset_index(drop=True)
    return df


def build_df(xml_files):
    # parse xml files and create pandas dataframe
    df = xml_files_to_df(xml_files)
    

    classes = sorted(df['names'].unique())
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    df['label'] = df['names'].map(cls2id)
    
    # in this df, each object of a given image is in a separate row
    df = df[['id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'img_path']]
    
    return df, classes