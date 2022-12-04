import glob
import sys
from math import tan, pi
import torch
import torchvision
import cv2

import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer

import shutil
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askdirectory
import csv
from openpyxl import Workbook

#TODO HOOKS
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime


# class LossEvalHook(HookBase):
#     def __init__(self, eval_period, model, data_loader):
#         self._model = model
#         self._period = eval_period
#         self._data_loader = data_loader
#
#     def _do_loss_eval(self):
#         # Copying inference_on_dataset from evaluator.py
#         total = len(self._data_loader)
#         num_warmup = min(5, total - 1)
#
#         start_time = time.perf_counter()
#         total_compute_time = 0
#         losses = []
#         for idx, inputs in enumerate(self._data_loader):
#             if idx == num_warmup:
#                 start_time = time.perf_counter()
#                 total_compute_time = 0
#             start_compute_time = time.perf_counter()
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             total_compute_time += time.perf_counter() - start_compute_time
#             iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
#             seconds_per_img = total_compute_time / iters_after_start
#             if idx >= num_warmup * 2 or seconds_per_img > 5:
#                 total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
#                 eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
#                 log_every_n_seconds(
#                     logging.INFO,
#                     "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
#                         idx + 1, total, seconds_per_img, str(eta)
#                     ),
#                     n=5,
#                 )
#             loss_batch = self._get_loss(inputs)
#             losses.append(loss_batch)
#         mean_loss = np.mean(losses)
#         self.trainer.storage.put_scalar('validation_loss', mean_loss)
#         comm.synchronize()
#
#         return losses
#
#     def _get_loss(self, data):
#         # How loss is calculated on train_loop
#         metrics_dict = self._model(data)
#         metrics_dict = {
#             k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
#             for k, v in metrics_dict.items()
#         }
#         total_losses_reduced = sum(loss for loss in metrics_dict.values())
#         return total_losses_reduced
#
#     def after_step(self):
#         next_iter = self.trainer.iter + 1
#         is_final = next_iter == self.trainer.max_iter
#         if is_final or (self._period > 0 and next_iter % self._period == 0):
#             self._do_loss_eval()
#         self.trainer.storage.put_scalars(timetest=12)



# TODO NOT FREEZE
def run():
    torch.multiprocessing.freeze_support()
    print('loop')

# TODO GETTING DATA
def get_data_dicts(directory, classes):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}

        filename = os.path.join(directory, img_anns["imagePath"])

        record["file_name"] = filename
        record["height"] = 800
        record["width"] = 800

        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]  # x coord
            py = [a[1] for a in anno['points']]  # y-coord
            poly = [(x, y) for x, y in zip(px, py)]  # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

if __name__ == '__main__':
    run()


# TODO CLASS AND SHIT
#classes = ['CR', 'IP', 'LF', 'PO', 'Sl', 'UC']
classes = ['signature', 'date']

data_path = 'C:/Users/kamra/Documents/My Documents/2022 Fall/Deep Learning/Project/ds_reformatted1/'

for d in ["train", "test"]:
    DatasetCatalog.register(
        "category_" + d,
        lambda d=d: get_data_dicts(data_path+d, classes)
    )
    MetadataCatalog.get("category_" + d).set(thing_classes=classes)

microcontroller_metadata = MetadataCatalog.get("category_train")


# TODO TRAIN OPTIONS
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
#cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
#cfg.MODEL.RESNETS.DEPTH = 34
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("category_train",)
cfg.DATASETS.TEST = ()
cfg.TEST.EVAL_PERIOD = 15
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 250
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

#TODO CUSTOM TRAINER
#custom trainer
# class MyTrainer(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         return COCOEvaluator(dataset_name, cfg, True, output_folder)
#
#     def build_hooks(self):
#         hooks = super().build_hooks()
#         hooks.insert(-1, LossEvalHook(
#             cfg.TEST.EVAL_PERIOD,
#             self.model,
#             build_detection_test_loader(
#                 self.cfg,
#                 self.cfg.DATASETS.TEST[0],
#                 DatasetMapper(self.cfg, True)
#             )
#         ))
#         return hooks

# TODO TRAIN
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()


# TODO TEST OPTIONS
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

predictor = DefaultPredictor(cfg)

# TODO TEST ONLY

import json
import matplotlib.pyplot as plt

#experiment_folder = './output/'

# def load_json_arr(json_path):
#     lines = []
#     with open(json_path, 'r') as f:
#         for line in f:
#             lines.append(json.loads(line))
#     return lines
#
#
# experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
#
# plt.plot(
#     [x['iteration'] for x in experiment_metrics],
#     [x['total_loss'] for x in experiment_metrics])
# plt.plot(
#     [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
#     [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
# plt.legend(['total_loss', 'validation_loss'], loc='upper left')
# plt.show()


for imageName in sorted(glob.glob('C:\\Users\\kamra\\Documents\\My Documents\\2022 Fall\\Deep Learning\\Project\\ds_reformatted1\\test\\*png')):
    im = cv2.imread(imageName)
    im2 = Image.open(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=microcontroller_metadata)
    #
    pred_classes = outputs['instances'].pred_classes.cpu().tolist()
    class_names = MetadataCatalog.get("category_train").thing_classes
    pred_class_names = list(map(lambda x: class_names[x], pred_classes))
    #Save detected image
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    v.save(imageName + "_test.png")
    # Cropping
    i = 0
    for pred in range(len(pred_class_names)):
        if pred_class_names[pred] == "signature":
            break
        else:
            i += 1
    boxes = list(outputs["instances"].pred_boxes)
    box = boxes[i]
    box = box.detach().cpu().numpy()
    x_top_left = box[0]
    y_top_left = box[1]
    x_bottom_right = box[2]
    y_bottom_right = box[3]
    x_center = (x_top_left + x_bottom_right) / 2
    y_center = (y_top_left + y_bottom_right) / 2
    crop_img = im2.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
    crop_img.save(imageName + "_cropped.png")

