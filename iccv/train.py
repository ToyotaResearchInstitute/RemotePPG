import os
import copy

import torch
import torch.optim as optim

from src.loss_initialization import init_loss
from src.model_loader import load_model
from src.parse_args import parse_args
from src.setup_dataloader import setup_dataloader
from src.shared.torch_utils import set_random_seed
from src.train_model import train_model
from src.test_model import test_model
from src.output_logger import OutputLogger


if __name__ == '__main__':
    # Train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Parse args
    args = parse_args('train')
    hyper_params = vars(args)

    # Fix random seed for reproducability
    set_random_seed(args.random_seed)

    # Create output dir
    if not args.output_dir:
        args.output_dir = 'checkpoints/temp'
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize output logger
    logger = OutputLogger(args.wandb_project_name is not None, args.output_dir)
    if args.wandb_project_name is not None:
        import wandb
        wandb.init(project=args.wandb_project_name,
                group=hyper_params['group_name'],
                name=hyper_params['exp_name'],
                tags=hyper_params['exp_tags'],
                config=hyper_params)

    # Dataset and dataloader construction
    trainloader = setup_dataloader(args, 'train')
    if args.val_on_train:
        devloader = None
    else:
        if args.val_on_test:
            if args.test_filter_activities is not None:
                args.filter_activities = args.test_filter_activities
            if args.test_datasets is not None:
                args.datasets = args.test_datasets
            if args.test_protocol is not None:
                args.protocol = args.test_protocol
            args.downsample = args.test_downsample
        devloader = setup_dataloader(args, 'dev')

    # Combine data loaders
    dataloaders = {'train': trainloader, 'val': devloader}
    Fs = trainloader.dataset.options.target_freq

    # Define PPG metrics and loss functions
    ppg_metrics = ['NegMCC', 'NegSNR', 'IPR', 'NegPC']
    loss_stat_names = []
    loss_funcs = {}
    for loss_name in args.loss:
        if loss_name not in ppg_metrics:
            ppg_metrics.append(loss_name)
        loss_stat_names.append('ppg_' + loss_name)
    for ppg_metric in ppg_metrics:
        loss_funcs[ppg_metric] = init_loss(ppg_metric, device, Fs, dataloaders['train'].dataset.options.D, args)
        loss_funcs[ppg_metric].to(device)

    # Skip if not training
    if args.epochs != 0:
        # Load models
        model = load_model(args.model, device, trainloader.dataset, args.pretrained_weights, args)
        if args.wandb_project_name is not None:
            wandb.watch(model)

        # Initialize optimizer
        adjusted_lr = args.lr * args.batch_size
        if hyper_params['optimizer'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=adjusted_lr)
        elif hyper_params['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=adjusted_lr, momentum=0.9)
        else:
            raise NotImplementedError

        # Optionally add learning rate scheduler
        scheduler_type = hyper_params['scheduler']
        if scheduler_type is None:
            print('Not using a learning rate scheduler.')
            scheduler = None
        else:
            print(f'Using {scheduler_type} scheduler.')
            params = hyper_params['scheduler_params']
            if scheduler_type == 'step':
                assert(len(params) == 2)
                scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=params[0],
                                                    gamma=params[1])
            elif scheduler_type == 'exponential':
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params[0])
            elif scheduler_type == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params[0])
            elif scheduler_type == 'cyclic':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=params[0], max_lr=params[1],
                                                        step_size_up=params[2], step_size_down=params[3])
            else:
                raise NotImplementedError

        # Start training
        train_model(model, dataloaders, optimizer, scheduler, ppg_metrics, loss_stat_names, loss_funcs, device, args, logger)
        print('\nTraining is finished without flaw!')

    # Override training parameters
    args.time_depth = args.test_time_depth
    if args.test_time_depth_extractor is not None:
        args.time_depth_extractor = args.test_time_depth_extractor
    if args.test_filter_activities is not None:
        args.filter_activities = args.test_filter_activities
    if args.test_datasets is not None:
        args.datasets = args.test_datasets
    if args.test_protocol is not None:
        args.protocol = args.test_protocol
    if args.test_batch_size is not None:
        args.batch_size = args.test_batch_size
    if args.test_exclude_border is not None:
        args.exclude_border = args.test_exclude_border
    args.downsample = args.test_downsample

    # Setup test dataset
    testloader = setup_dataloader(args, 'test')

    # Loop through each validation metric used
    wandb_step = args.epochs + 10
    metric_names = args.val_metric + ['Last',]
    for metric_name in metric_names:
        # Reload model
        weights_path = os.path.join(args.output_dir, f'model_{metric_name}.pt') if args.epochs != 0 else args.pretrained_weights
        model = load_model(args.model, device, trainloader.dataset, weights_path, args)

        # Start testing
        wandb_step = test_model(model, testloader, metric_name, ppg_metrics, loss_funcs, device, wandb_step, args, logger)

    # Finalize log
    logger.finalize()
    print('\nTesting finished without flaw!')
