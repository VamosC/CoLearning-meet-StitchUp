import mmcv
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from mllt.datasets.registry import DATASETS
from mllt.datasets.custom import CustomDataset
import random


@DATASETS.register_module
class VOCNoiseDataset(CustomDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, noise_label_file, **kwargs):
        super(VOCNoiseDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
        self.categories = self.CLASSES
        self.noise_label_file = noise_label_file
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.anns = mmcv.load(noise_label_file)
        self.index_dic = self.get_index_dic()
        # self.head_list = list(self.class_split['head'])
        # self.medium_list = list(self.class_split['middle'])
        # self.tail_list = list(self.class_split['tail'])

    def load_annotations(self, ann_file, LT_ann_file=None):
        img_infos = []
        self.img_ids = mmcv.list_from_file(ann_file)
        for img_id in self.img_ids:
            filename = 'JPEGImages/{}.jpg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):

        labels = self.anns['labels'][idx]
        clean_labels = self.anns['clean_labels'][idx]
        ann = dict(labels=labels,
                   clean_labels=clean_labels)
        return ann

    def __getitem__(self, idx):

        cls_id = None

        if isinstance(idx, tuple):
            cls_id = idx[1]
            idx = idx[0]

        data = super().__getitem__(idx)
        if self.test_mode:
            return data

        label = data['noisy_labels']
        if cls_id is None:
            cls_ids = np.where(label > 0)[0]
            cls_id = np.random.choice(cls_ids, 1).item()
        cls_list = self.index_dic[cls_id]

        data1 = self.prepare_train_img(np.random.choice(cls_list, 1).item())

        return data, data1
