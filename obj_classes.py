import os
from os import listdir
from os.path import isfile,join
import sys
import getopt
import ijson
from numpy.testing._private.utils import break_cycles
import wget
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from src.model import EfficientDet

from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.autonotebook import tqdm
import cv2
from configparser import ConfigParser
file = 'config.ini'
config = ConfigParser()
config.read(file)

class obj_det():
    def __init__(self):
        pass

    def data_processing(self):
        fd = open(config['training_info']['train_json_file'],'r')
        objs = ijson.items(fd, 'categories.item')
        labels = (o for o in objs)
        self.class_count = 0
        dict_list = {}
        classes = []
        for label in labels:
            # print('id:{}, category:{}, super category:{}'.format(label['id'], label['name'], label['supercategory']))
            dict_list[label['id']] = label['name']
            self.class_count += 1

        for i in sorted(dict_list):
            classes.append(dict_list[i])
        # print('Total categories/labels: ', len(classes))
        # print(dict_list)
        # print(classes)

        fd.close()

        self.COCO_CLASSES = classes
        # print(self.COCO_CLASSES)

        self.colors = [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86), (14, 89, 122),
                (80, 7, 65), (10, 102, 25), (90, 185, 109), (106, 110, 132), (169, 158, 85), (188, 185, 26), (103, 1, 17),
                (82, 144, 81), (92, 7, 184), (49, 81, 155), (179, 177, 69), (93, 187, 158), (13, 39, 73), (12, 50, 60),
                (16, 179, 33), (112, 69, 165), (15, 139, 63), (33, 191, 159), (182, 173, 32), (34, 113, 133), (90, 135, 34),
                (53, 34, 86), (141, 35, 190), (6, 171, 8), (118, 76, 112), (89, 60, 55), (15, 54, 88), (112, 75, 181),
                (42, 147, 38), (138, 52, 63), (128, 65, 149), (106, 103, 24), (168, 33, 45), (28, 136, 135), (86, 91, 108),
                (52, 11, 76), (142, 6, 189), (57, 81, 168), (55, 19, 148), (182, 101, 89), (44, 65, 179), (1, 33, 26),
                (122, 164, 26), (70, 63, 134), (137, 106, 82), (120, 118, 52), (129, 74, 42), (182, 147, 112), (22, 157, 50),
                (56, 50, 20), (2, 22, 177), (156, 100, 106), (21, 35, 42), (13, 8, 121), (142, 92, 28), (45, 118, 33),
                (105, 118, 30), (7, 185, 124), (46, 34, 146), (105, 184, 169), (22, 18, 5), (147, 71, 73), (181, 64, 91),
                (31, 39, 184), (164, 179, 33), (96, 50, 18), (95, 15, 106), (113, 68, 54), (136, 116, 112), (119, 139, 130),
                (31, 139, 34), (66, 6, 127), (62, 39, 2), (49, 99, 180), (49, 119, 155), (153, 50, 183), (125, 38, 3),
                (129, 87, 143), (49, 87, 40), (128, 62, 120), (73, 85, 148), (28, 144, 118), (29, 9, 24), (175, 45, 108),
                (81, 175, 64), (178, 19, 157), (74, 188, 190), (18, 114, 2), (62, 128, 96), (21, 3, 150), (0, 6, 95),
                (2, 20, 184), (122, 37, 185)]

        

    def train(self):    
        image_size = int(config['training_info']['image_size'])
        batch_size = int(config['training_info']['batch_size'])
        lr = 1e-4
        alpha = 0.25
        gamma = 1.5
        num_epochs = int(config['training_info']['num_epochs'])
        test_interval = 1
        es_min_delta = 0.0
        es_patience = 0
        data_path = config['training_info']['data_path']
        log_path = "tensorboard/signatrix_efficientdet_coco"
        saved_path = config['training_info']['saved_path']
        num_gpus = 1
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        training_params = {"batch_size": batch_size * num_gpus,
                        "shuffle": True,
                        "drop_last": True,
                        "collate_fn": collater,
                        "num_workers": 12}

        test_params = {"batch_size": batch_size,
                    "shuffle": False,
                    "drop_last": False,
                    "collate_fn": collater,
                    "num_workers": 12}

        training_set = CocoDataset(root_dir=data_path, set="train2017",
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        training_generator = DataLoader(training_set, **training_params)

        test_set = CocoDataset(root_dir=data_path, set="val2017",
                            transform=transforms.Compose([Normalizer(), Resizer()]))
        test_generator = DataLoader(test_set, **test_params)

        model = EfficientDet(num_classes = len(self.COCO_CLASSES))


        if os.path.isdir(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)

        if not os.path.isdir(saved_path):
            os.makedirs(saved_path)

        writer = SummaryWriter(log_path)
        if torch.cuda.is_available():
            model = model.cuda()
            model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

        best_loss = 1e5
        best_epoch = 0
        model.train()

        num_iter_per_epoch = len(training_generator)
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                try:
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                    else:
                        cls_loss, reg_loss = model([data['img'].float(), data['annot']])

                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                    loss = cls_loss + reg_loss
                    if loss == 0:
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    epoch_loss.append(float(loss))
                    total_loss = np.mean(epoch_loss)

                    progress_bar.set_description(
                        'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                            epoch + 1, num_epochs, iter + 1, num_iter_per_epoch, cls_loss, reg_loss, loss,
                            total_loss))
                    writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                    writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
                    writer.add_scalar('Train/Classfication_loss (focal loss)', cls_loss, epoch * num_iter_per_epoch + iter)

                except Exception as e:
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch %  test_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(test_generator):
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                        else:
                            cls_loss, reg_loss = model([data['img'].float(), data['annot']])

                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss_classification_ls.append(float(cls_loss))
                        loss_regression_ls.append(float(reg_loss))

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch + 1, num_epochs, cls_loss, reg_loss,
                        np.mean(loss)))
                writer.add_scalar('Test/Total_loss', loss, epoch)
                writer.add_scalar('Test/Regression_loss', reg_loss, epoch)
                writer.add_scalar('Test/Classfication_loss (focal loss)', cls_loss, epoch)

                if loss + es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    torch.save(model, os.path.join(saved_path, "signatrix_efficientdet_coco.pth"))

                    dummy_input = torch.rand(batch_size, 3, 512, 512)
                    if torch.cuda.is_available():
                        dummy_input = dummy_input.cuda()
                    if isinstance(model, nn.DataParallel):
                        model.module.backbone_net.model.set_swish(memory_efficient=False)

                        torch.onnx.export(model.module, dummy_input,
                                        os.path.join(saved_path, "signatrix_efficientdet_coco.onnx"),
                                        verbose=False)
                        model.module.backbone_net.model.set_swish(memory_efficient=True)
                    else:
                        model.backbone_net.model.set_swish(memory_efficient=False)

                        torch.onnx.export(model, dummy_input,
                                        os.path.join(saved_path, "signatrix_efficientdet_coco.onnx"),
                                        verbose=False)
                        model.backbone_net.model.set_swish(memory_efficient=True)

                # Early stopping
                if epoch - best_epoch > es_patience > 0:
                    print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                    break
        writer.close()


    def test_images(self):
        image_size = int(config['test_info_images']['image_size'])
        test_imgs = config['test_info_images']['test_imgs']
        cls_threshold = float(config['test_info_images']['cls_threshold'])
        pretrained_model = config['test_info_images']['pretrained_model']
        output = config['test_info_images']['output_path']
        nms_threshold = 0.5



        model = torch.load( pretrained_model).module
        model.cuda()
        imgs_paths = sorted([os.path.abspath(os.path.join(test_imgs, p)) 
									for p in os.listdir(test_imgs)])

        onlyfiles = sorted([f for f in listdir(test_imgs) if isfile(join(test_imgs, f))])
        if os.path.isdir(output):
            shutil.rmtree(output)
        os.makedirs(output)

        for index in range(len(imgs_paths)):
            image = cv2.imread(imgs_paths[index])
            output_image = np.copy(image)
            # print(image)
            if image.size != 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                break
            height, width = image.shape[:2]
            image = image.astype(np.float32) / 255
            image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
            if height > width:
                scale = image_size / height
                resized_height = image_size
                resized_width = int(width * scale)
            else:
                scale = image_size / width
                resized_height = int(height * scale)
                resized_width = image_size

            image = cv2.resize(image, (resized_width, resized_height))

            new_image = np.zeros((image_size, image_size, 3))
            new_image[0:resized_height, 0:resized_width] = image
            new_image = np.transpose(new_image, (2, 0, 1))
            new_image = new_image[None, :, :, :]
            new_image = torch.Tensor(new_image)
            if torch.cuda.is_available():
                new_image = new_image.cuda()
            with torch.no_grad():
                scores, labels, boxes = model(new_image)
                boxes /= scale
            if boxes.shape[0] == 0:
                continue

            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < cls_threshold:
                    break
                pred_label = int(labels[box_id])
                xmin, ymin, xmax, ymax = boxes[box_id, :]
                color = self.colors[pred_label]
                xmin = int(xmin);ymin = int(ymin);xmax = int(xmax);ymax = int(ymax)
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(self.COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, self.COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)

                cv2.imwrite("{}/{}_prediction.jpg".format(output, onlyfiles[index]), output_image)




    def test_video(self):
        image_size = int(config['test_info_video']['image_size'])
        cls_threshold = float(config['test_info_video']['cls_threshold'])
        pretrained_model = config['test_info_video']['pretrained_model']
        
        input = config['test_info_video']['input_video']
        # print(input)
        output = config['test_info_video']['output_path']
        nms_threshold = 0.5
        model = torch.load(pretrained_model).module
        if torch.cuda.is_available():
            model.cuda()

        cap = cv2.VideoCapture(input)
        out = cv2.VideoWriter(output,  cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        while cap.isOpened():
            flag, image = cap.read()
            output_image = np.copy(image)
            # print(image)
            if flag:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                break
            height, width = image.shape[:2]
            image = image.astype(np.float32) / 255
            image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
            if height > width:
                scale = image_size / height
                resized_height = image_size
                resized_width = int(width * scale)
            else:
                scale = image_size / width
                resized_height = int(height * scale)
                resized_width = image_size

            image = cv2.resize(image, (resized_width, resized_height))

            new_image = np.zeros((image_size, image_size, 3))
            new_image[0:resized_height, 0:resized_width] = image
            new_image = np.transpose(new_image, (2, 0, 1))
            new_image = new_image[None, :, :, :]
            new_image = torch.Tensor(new_image)
            if torch.cuda.is_available():
                new_image = new_image.cuda()
            with torch.no_grad():
                scores, labels, boxes = model(new_image)
                boxes /= scale
            if boxes.shape[0] == 0:
                continue

            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < cls_threshold:
                    break
                pred_label = int(labels[box_id])
                xmin, ymin, xmax, ymax = boxes[box_id, :]
                color = self.colors[pred_label]
                xmin = int(xmin);ymin = int(ymin);xmax = int(xmax);ymax = int(ymax)
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(self.COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, self.COCO_CLASSES[pred_label] + ' : %.2f' % pred_prob,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
            out.write(output_image)

        cap.release()
        out.release()