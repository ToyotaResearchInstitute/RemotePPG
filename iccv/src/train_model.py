import random
import numpy as np
from collections import OrderedDict
import copy

import torch

from src.shared.heart_rate import predict_heart_rate
from src.shared.train_utils import *


def train_model(model, dataloaders, optimizer, scheduler, ppg_metrics, loss_stat_names, loss_funcs, device, args, logger):
    # set up "best" validation metrics for model saving
    Fs = dataloaders['train'].dataset.options.target_freq
    best_val_metrics = args.val_metric
    assert(isinstance(best_val_metrics, list))
    min_val_losses = {}
    for metric in best_val_metrics:
        min_val_losses[metric] = None

    # Loop through all epochs
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        phases = ['train']
        if not args.val_on_train:
            phases.append('val')
        for phase in phases:
            # Setup stats
            epoch_hr = {'gt': [], 'our': [], 'pred': []}
            running_stats = OrderedDict()
            if phase == 'train':
                running_stats['loss'] = 0.0
            for ppg_metric in ppg_metrics:
                running_stats['ppg_' + ppg_metric] = 0.0
            
            # Set model to train or evaluate
            if phase == 'train':
                model.train()  # Set model to training mode -> activate droput layers and batch norm
            else:
                model.eval()  # Set model to evaluate mode

            # Determine which batch to grab.
            num_batches = len(dataloaders[phase])
            grab_batch = random.randrange(num_batches)

            # Iterate over data.
            for batch_on, samples in enumerate(dataloaders[phase]):
                # Grab a set of training videos.
                num_samples = samples['video'].size(0)
                plot_sample = random.randrange(num_samples)
                if batch_on == grab_batch:
                    log_video = samples['video'][plot_sample].detach().clone().numpy()
                    logger.output_video(log_video, epoch, Fs, phase)

                # Send video to the device.
                samples['video'] = samples['video'].to(device)

                # Send labels to the device.
                for key, target in samples['targets'].items():
                    samples['targets'][key] = target.to(device)

                # Trim the borders of the targets
                for key, target in samples['targets'].items():
                    if args.exclude_border != 0:
                        samples['targets'][key] = target[:,args.exclude_border:-args.exclude_border]

                # Forward pass - get the signals
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model output
                    signals, batch_stats, ss_out = get_model_output(model, args.model, samples['video'], loss_funcs, args)

                    # Calculate PPG stats
                    new_batch_stats = calculate_ppg_metrics(signals, samples['targets'], ppg_metrics, loss_funcs)
                    batch_stats.update(new_batch_stats)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss = 0.0
                    for i, loss_stat_name in enumerate(loss_stat_names):
                        loss += batch_stats[loss_stat_name] * args.loss_weights[i]
                    running_stats['loss'] += loss.item()
                    if args.ss:
                        _, t, b, g1, g2 = ss_out[1].shape
                        # apply sparsity loss 
                        if args.ss_sparsity > 0:
                            entropy = -torch.sum(torch.log(ss_out[1].view(-1)) * ss_out[1].view(-1))
                            loss += entropy * args.ss_sparsity / torch.numel(ss_out[1])
                        # apply temporal consistency loss
                        if args.ss_temporal > 0:
                            ss_diff = ss_out[1][:,1:,:,:,:] - ss_out[1][:,:-1,:,:,:]
                            ssd = torch.sum(ss_diff.view(-1) * ss_diff.view(-1))
                            loss += ssd * args.ss_temporal / torch.numel(ss_out[1])

                    loss.backward()
                    optimizer.step()

                # Grab output and ground truth.
                gt_values = {}
                for key, target in samples['targets'].items():
                    gt_values[key] = target.cpu().numpy()
                gt_signals = gt_values['ppg']
                pred_signals = signals.cpu().detach().numpy()

                # Predict whole-sample heart rate
                batch_hr_gt = np.mean(gt_values['rate'], -1)
                batch_hr_our = np.mean(gt_values['our'], -1)
                batch_hr_pred = []
                for i in range(num_samples):
                    sample_hr = predict_heart_rate(pred_signals[i], Fs, args.high_pass, args.low_pass)
                    batch_hr_pred.append(sample_hr)

                # Grab a sample validation output plot.
                if batch_on == grab_batch:
                    logger.output_ppg(samples['id_time'][plot_sample], gt_signals[plot_sample], pred_signals[plot_sample], 
                            batch_hr_gt[plot_sample], batch_hr_our[plot_sample], batch_hr_pred[plot_sample],
                            epoch, phase, Fs)
                if batch_on == grab_batch and (phase == 'val' or args.val_on_train) and args.ss:
                    logger.output_video(ss_out[0][plot_sample].detach().cpu().numpy(), epoch, Fs, vid_name="resampled")
                    saliency_maps = ss_out[1][plot_sample].permute(1,0,2,3).detach().cpu().numpy()
                    saliency_maps *= 1/np.max(saliency_maps)
                    logger.output_video(saliency_maps, epoch, Fs, vid_name="saliency")

                # Log batch statistics
                for batch_stat_name, batch_stat in batch_stats.items():
                    if batch_stat_name not in running_stats:
                        running_stats[batch_stat_name] = 0.0
                    running_stats[batch_stat_name] += batch_stat.item() * num_samples
                epoch_hr['gt'].extend(batch_hr_gt)
                epoch_hr['our'].extend(batch_hr_our)
                epoch_hr['pred'].extend(batch_hr_pred)

            # Aggregate and print epoch stats
            stats, epoch_hr = aggregate_epoch_stats(running_stats, epoch_hr, phase)

            # Log epoch stats
            logger.output_epoch_stats(stats, epoch_hr, epoch, phase)
            
            # log the learning rate
            assert(len(optimizer.param_groups) == 1)
            logger.output_values({'learning_rate': optimizer.param_groups[0]['lr']}, step = epoch)

            # Save best / last model
            if phase == 'val' or args.val_on_train:
                # Always save model at final epoch by default
                if epoch == args.epochs-1:
                    checkpoint_path = f'{args.output_dir}/model_Last.pt'
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.save(checkpoint_path)
                # Save best model(s) according to best_val_metrics
                for metric in best_val_metrics:
                    min_val_loss = min_val_losses[metric]
                    if not np.isnan(stats[metric]):
                        if min_val_loss is None or stats[metric] < min_val_loss:
                            min_val_losses[metric] = stats[metric]
                            checkpoint_path = f'{args.output_dir}/model_{metric}.pt'
                            torch.save(model.state_dict(), checkpoint_path)
                            logger.save(checkpoint_path)
                            print(f'New best validation loss ({metric}): {min_val_losses[metric]}')
        
        # Advance the LR scheduler
        print()
        if scheduler is not None:
            scheduler.step()
