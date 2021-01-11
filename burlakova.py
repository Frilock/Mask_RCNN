from numpy import expand_dims

from mrcnn.model import MaskRCNN
from mrcnn.config import Config


class TrafficSignConfig(Config):
    NAME = "TrafficSign"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2


config = TrafficSignConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=config)
model.load_weights('traffic_sign.h5', by_name=True)


def burlakova(image):
    sample = expand_dims(image, 0)  # добавляет новую ось к массиву
    yhat = model.detect(sample, verbose=0)[0]
    boxes = []
    for box in yhat['rois']:
        y1, x1, y2, x2 = box
        boxes.append(
            {
                'bbox':
                    {
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                    }
            }
        )
    return boxes
