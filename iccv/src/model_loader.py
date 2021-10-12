import torch

from src.archs.FrequencyContrast import FrequencyContrast
from src.archs.SaliencyNet import saliency_network_resnet18
from src.archs.SaliencySampler import SaliencySampler
from src.model_initialization import init_model


def load_model(model_name, device, dataset, weights_path, args):
    # Initialize models
    if model_name == 'FrequencyContrast':
        model = FrequencyContrast(args, device, dataset)
    else:
        model = init_model(model_name, args, device, dataset)

    # Add optional saliency sampler to model
    if args.ss:
        use_pretrained = True if args.ss_pretrain == 1 else False
        saliency_net_depth = args.ss_layers
        
        if args.ss_out_dim is None or len(args.ss_out_dim) == 0:
            task_input_size = (args.scale_height, args.scale_width)
        elif len(args.ss_out_dim) == 1:
            task_input_size = (args.ss_out_dim, args.ss_out_dim)
        else:
            assert(len(args.ss_out_dim) == 2)
            task_input_size = (args.ss_out_dim[0], args.ss_out_dim[1])

        if len(args.ss_dim) == 1:
            saliency_input_size = (args.ss_dim, args.ss_dim)
        else:
            assert(len(args.ss_dim) == 2)
            saliency_input_size = (args.ss_dim[0], args.ss_dim[1])

        channels = 3 if args.use_channels is None else len(args.use_channels)
        print(channels, "channels")
        saliency_network = saliency_network_resnet18(device,
                                                     use_pretrained,
                                                     saliency_net_depth,
                                                     channels)
        model = SaliencySampler(model,
                                saliency_network,
                                saliency_net_depth,
                                device,
                                task_input_size=task_input_size,
                                saliency_input_size=saliency_input_size)

    # Try loading pretrained weights before DataParallel.
    loaded_weights = False
    if weights_path is not None:
        try:
            model.load_state_dict(torch.load(weights_path))
            loaded_weights = True
            print(f'Pre-trained weights are loaded for model {model_name}')
        except:
            pass

    # Use multiple GPU if there are!
    if device != torch.device('cpu') and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Try loading pretrained weights after DataParallel.
    if weights_path is not None and not loaded_weights:
        model.load_state_dict(torch.load(weights_path))
        print(f'Pre-trained weights are loaded for model {model_name}')

    # Copy model to working device
    model = model.to(device)
    return model