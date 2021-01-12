import models

avaliable_models = [
    'vgg16',
    'vgg16-baseline',
    'efficientnetb0',
    'efficientnetb0-baseline'
]

models_dict = {
    'vgg16': (models.VGG16, models.process_vgg16),
    'vgg16-baseline': (models.VGG16_baseline, models.process_vgg16),
    'efficientnetb0': (models.EfficientNetB0, models.process_efficientnetb0),
    'efficientnetb0-baseline': (models.EfficientNetB0_baseline, models.process_efficientnetb0)
}


def get_feature_extractor(model):
    (modelfn, fn) = models_dict[model]
    return modelfn(), fn
