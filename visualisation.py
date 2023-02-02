from torchvision import models

path = 'models/'
model_refs = {
    'ResNet18': {'model':models.resnet18, 'path': '20230131083336resnet18'},
    'ResNet34': {'model':models.resnet34, 'path': '20230131083251resnet34'},
    'ResNet50': {'model':models.resnet50, 'path': '20230131083353resnet50'},
}
