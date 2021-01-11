import json

from os import listdir
from numpy import zeros
from numpy import asarray
from PIL import Image

from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

image_dir = "C://Users//Ivan//PyCharmProjects//Mask_RCNN//images//trafficSignDataset//"
h5_directory = "mask_rcnn_coco.h5"
save_model_path = "traffic_sign.h5"


class TrafficSignConfig(Config):
    NAME = "TrafficSign"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 2


class TrafficSignDataset(Dataset):
    def load_dataset(self, dataset_dir, train_part=0.75, total_im=2000, is_train=True):
        self.add_class("dataset", 1, "trafficsign")

        matching = [s for s in listdir(dataset_dir) if "jpg" in s]
        train_num = int(total_im * train_part)

        if is_train:
            images_set = matching[:train_num]
        else:
            images_set = matching[train_num:total_im]

        for filename in images_set:
            # извлечение идентификатора картинки
            image_id = filename[:-3]

            image_path = dataset_dir + filename
            annotation_path = dataset_dir + filename[:-3] + 'json'

            self.add_image('dataset', image_id=image_id, path=image_path, annotation=annotation_path)

    @staticmethod
    def extract_boxes(filename):
        with open(filename) as file:
            config_str = file.read()

        json_config = json.loads(config_str)
        # выделение рамок (ограничивающего прямоугольника)
        boxes = list()
        for i in json_config:
            x_min = int(i['bbox']['xmin'])
            y_min = int(i['bbox']['ymin'])
            x_max = int(i['bbox']['xmax'])
            y_max = int(i['bbox']['ymax'])
            coors = [x_min, y_min, x_max, y_max]
            boxes.append(coors)

        image = Image.open(filename[:-4] + 'jpg')
        width, height = image.size
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        # определение расположения файла
        path = info['annotation']
        boxes, width, height = self.extract_boxes(path)
        # создание одного массива для всех массок, каждый на своем канале
        masks = zeros([height, width, len(boxes)], dtype='uint8')
        # создание масок
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('trafficsign'))
        return masks, asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


train_dataset = TrafficSignDataset()
train_dataset.load_dataset(image_dir, is_train=True)
train_dataset.prepare()

validation_dataset = TrafficSignDataset()
validation_dataset.load_dataset(image_dir, is_train=False)
validation_dataset.prepare()

config = TrafficSignConfig()

model = MaskRCNN(mode="training", config=config, model_dir='./')
model.load_weights(filepath=h5_directory,
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model.train(train_dataset=train_dataset,
            val_dataset=validation_dataset,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            layers='heads')
model.keras_model.save_weights(save_model_path)
