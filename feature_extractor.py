import models

avaliable_models = [
    'vgg16',
    'vgg16-baseline',
    'efficientnetb0',
    'efficientnetb0-baseline',
    'vgg19',
    'vgg19-baseline',
    'inceptionv3',
    'inceptionv3-baseline'
]

models_dict = {
    'vgg16': (models.VGG16, models.process_vgg16),
    'vgg16-baseline': (models.VGG16_baseline, models.process_vgg16),
    'efficientnetb0': (models.EfficientNetB0, models.process_efficientnetb0),
    'efficientnetb0-baseline': (models.EfficientNetB0_baseline, models.process_efficientnetb0),
    'vgg19': (models.VGG19, models.process_vgg19),
    'vgg19-baseline': (models.VGG19_baseline, models.process_vgg19),
    'inceptionv3': (models.InceptionV3, models.process_inception_v3),
    'inceptionv3-baseline': (models.InceptionV3_baseline, models.process_inception_v3)
}


def get_feature_extractor(model):
    (modelfn, fn) = models_dict[model]
    return modelfn(), fn
