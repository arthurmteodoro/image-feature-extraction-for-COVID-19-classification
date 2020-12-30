import models

avaliable_models = [
    'vgg16',
    'efficientnetb0'
]

models_dict = {
    'vgg16': (models.VGG16, models.process_vgg16),
    'efficientnetb0': (models.EfficientNetB0, models.process_efficientnetb0)
}


def get_feature_extractor(model):
    (modelfn, fn) = models_dict[model]
    return modelfn(), fn
