from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from .dataset_tools.coco import COCO
from .dataset_tools.kitti import KITTI
from .dataset_tools.coco_hp import COCOHP
from .dataset_tools.mot import MOT
from .dataset_tools.nuscenes import nuScenes
from .dataset_tools.crowdhuman import CrowdHuman
from .dataset_tools.kitti_tracking import KITTITracking
from .dataset_tools.custom_dataset import CustomDataset

dataset_factory = {
  'custom': CustomDataset,
  'coco': COCO,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'mot': MOT,
  'nuscenes': nuScenes,
  'crowdhuman': CrowdHuman,
  'kitti_tracking': KITTITracking,
}


def get_dataset(dataset):
  return dataset_factory[dataset]
