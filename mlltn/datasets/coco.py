import numpy as np
from pycocotools.coco import COCO
from mllt.datasets.custom import CustomDataset
from mllt.datasets.registry import DATASETS
import mmcv


@DATASETS.register_module
class CocoNoiseDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self, noise_label_file, **kwargs):
        super(CocoNoiseDataset, self).__init__(**kwargs)
        self.noise_label_file = noise_label_file
        self.anns = mmcv.load(noise_label_file)
        self.index_dic = self.get_index_dic()
        # self.head_list = list(self.class_split['head'])
        # self.medium_list = list(self.class_split['middle'])
        # self.tail_list = list(self.class_split['tail'])

    def load_annotations(self, ann_file, LT_ann_file=None):

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        self.categories = self.cat_ids  # cat_ids for coco and cat_names for voc
        if LT_ann_file is not None:
            self.img_ids = []
            for LT_ann_file in LT_ann_file:
                self.img_ids += mmcv.list_from_file(LT_ann_file)
        else:
            self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([int(i)])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):

        labels = self.anns['labels'][idx]
        clean_labels = self.anns['clean_labels'][idx]
        ann = dict(labels=labels,
                   clean_labels=clean_labels)
        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

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
