import models

avaliable_models = [
    'vgg16'
]


def get_feature_extractor(model):
    if model == 'vgg16':
        vgg16 = models.VGG16()
        return vgg16, models.process_vgg16
