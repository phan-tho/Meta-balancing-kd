from . import resnet

model_dict = {
    'resnet8': resnet.resnet8,
    'resnet14': resnet.resnet14,
    'resnet20': resnet.resnet20,
    'resnet32': resnet.resnet32,
    'resnet44': resnet.resnet44,
    'resnet56': resnet.resnet56,
    'resnet110': resnet.resnet110,
    'resnet8x4': resnet.resnet8x4,
    'resnet8x4_double': resnet.resnet8x4_double,
    'resnet32x4': resnet.resnet32x4,
    'resnet56x4': resnet.resnet56x4,
    'resnet110x4': resnet.resnet110x4,
}

def get_model(name, **kwargs):
    if name not in model_dict:
        raise KeyError(f"Model '{name}' not found")
    return model_dict[name](**kwargs)
