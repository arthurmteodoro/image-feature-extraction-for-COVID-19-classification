from . import models


def get_feature_extractor(model):
    if model == 'VGG16':
        vgg16 = models.VGG16()
        return vgg16, models.process_vgg16
