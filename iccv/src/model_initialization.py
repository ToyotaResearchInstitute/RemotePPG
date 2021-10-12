from src.archs.PhysNetUpsample import PhysNetUpsample
from src.archs.MeanBaseline import MeanBaseline
from src.archs.MedianBaseline import MedianBaseline


def init_model(model_name, args, device, dataset):
    if model_name == 'PhysNet':
        return PhysNetUpsample(dataset.options.C, args)
    elif model_name == 'MeanBaseline':
        return MeanBaseline(dataset)
    elif model_name == 'MedianBaseline':
        return MedianBaseline(dataset)
    else:
        print('ERROR: No such model. Choose from: PhysNet, MeanBaseline, MedianBaseline')
        exit(666)
