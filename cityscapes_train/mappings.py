import keras_cv

BACKBONE_DICT = {
    "resnet50_v2_imagenet": keras_cv.models.ResNet50V2Backbone,
    "resnet50_imagenet": keras_cv.models.ResNetBackbone,
    "mobilenet_v3_large_imagenet": keras_cv.models.MobileNetV3Backbone,
    "efficientnetv2_b0_imagenet": keras_cv.models.EfficientNetV2Backbone,
    "efficientnetv2_b1_imagenet": keras_cv.models.EfficientNetV2Backbone,
    "efficientnetv2_b2_imagenet": keras_cv.models.EfficientNetV2Backbone,
    "efficientnetv2_s_imagenet": keras_cv.models.EfficientNetV2Backbone,
}
