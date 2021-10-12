import numpy as np
from collections import OrderedDict
from scipy import signal

import torch
import torch.nn.functional as F

from src.shared.heart_rate import predict_heart_rate
from src.shared.train_utils import *


def test_model(model, dataloader, metric_name, ppg_metrics, loss_funcs, device, wandb_step, args, logger):   
    # Setup stats
    Fs = dataloader.dataset.options.target_freq
    running_stats = OrderedDict()
    for ppg_metric in ppg_metrics:
        running_stats['ppg_' + ppg_metric] = 0.0

    # Set model to evaluate mode
    model.eval()
    metric_output_name = "test_" + metric_name

    # Iterate over data.
    all_hr = {'gt': [], 'our': [], 'pred': []}
    for batch_on, samples in enumerate(dataloader):
        # Send data to the device.
        num_samples = samples['video'].size(0)
        samples['video'] = samples['video'].to(device)
        for key, target in samples['targets'].items():
            samples['targets'][key] = target.to(device)

        # Pad the temporal dimension with zeros to have an even number of window_offset.
        time_depth = samples['video'].size(2)
        if args.time_depth_extractor == 0:
            batch_time_depth_extractor = time_depth
            window_offset = 0
        else:
            batch_time_depth_extractor = args.time_depth_extractor
            window_offset = int(batch_time_depth_extractor // 2)
            padded_length = int(np.ceil(time_depth / window_offset)) * window_offset
            samples['video'] = F.pad(samples['video'], (0, 0, 0, 0, 0, padded_length - time_depth))
            for key, target in samples['targets'].items():
                samples['targets'][key] = F.pad(target, (0, padded_length - time_depth))
            time_depth = padded_length

        # Forward pass to get all batch PPG signals
        section_hrs = [[] for x in range(num_samples)]
        with torch.set_grad_enabled(False):
            signals = torch.zeros((num_samples, time_depth), device=device)
            window_totals = torch.zeros((1, time_depth), device=device)
            window_func = torch.hamming_window(batch_time_depth_extractor - (args.exclude_border * 2), device=device)
            for start_index in range(0, time_depth - window_offset, window_offset):
                # Take subset of video
                end_index = start_index + batch_time_depth_extractor
                video_subset = samples['video'][:,:,start_index:end_index]

                # Get model output
                section_signals, _, __ = get_model_output(model, args.model, video_subset, loss_funcs, args)

                # Window signal section and add to overall output
                signals[:,start_index+args.exclude_border:end_index-args.exclude_border] += section_signals * window_func
                window_totals[:,start_index+args.exclude_border:end_index-args.exclude_border] += window_func
            
                # Get the section heart rates
                section_signals = section_signals.cpu().detach().numpy()
                for i in range(num_samples):
                    if start_index == 0 or end_index < samples['length'][i]:
                        pred_hr = predict_heart_rate(section_signals[i], Fs, args.high_pass, args.low_pass)
                        section_hrs[i].append(pred_hr)

            # Get the final signals, weighted by window amounts
            signals /= window_totals

        # Loop through each sample
        for i in range(num_samples):
            # Get the subset of the signal and targets used
            num_frames = samples['length'][i]
            signal_pred = signals[i,:num_frames].unsqueeze(0)
            if args.exclude_border != 0:
                signal_pred = signal_pred[:,args.exclude_border:-args.exclude_border]
            gt_values = {}
            for key, target in samples['targets'].items():
                gt_values[key] = target[None,i,:num_frames]
                if args.exclude_border != 0:
                    gt_values[key] = gt_values[key][:,args.exclude_border:-args.exclude_border]

            # Calculate ppg loss stats
            sample_stats = calculate_ppg_metrics(signal_pred, gt_values, ppg_metrics, loss_funcs)
            for key, value in sample_stats.items():
                running_stats[key] += value

            # Move output values to CPU
            pred_signals = signal_pred.squeeze(0).cpu().detach().numpy()
            for key, target in gt_values.items():
                gt_values[key] = target.squeeze(0).cpu().detach().numpy()
            gt_signals = gt_values['ppg']
            
            # Estimate whole-sample heart rate
            hr_pred = predict_heart_rate(pred_signals, Fs, args.high_pass, args.low_pass)
            hr_rate = np.mean(gt_values['rate'])
            hr_our = np.mean(gt_values['our'])
            all_hr['gt'].append(hr_rate)
            all_hr['our'].append(hr_our)
            all_hr['pred'].append(hr_pred)

            # Output PPG and PSD
            logger.output_ppg(samples['id_time'][i], gt_signals, pred_signals, hr_rate, hr_our, hr_pred, wandb_step, metric_output_name, Fs)
            wandb_step += 1

        # save first N test video clips (where N = args.save_test_videos)
        if args.save_test_videos is not None and (batch_on % args.save_test_videos) == 0:
            for plot_sample in range(samples['video'].shape[0]):
                test_video_clip = samples['video'][plot_sample]
                logger.output_video(test_video_clip.detach().cpu().numpy(), wandb_step, Fs, metric_output_name)
                if args.ss:
                    _, _, ss_out = get_model_output(model, args.model, test_video_clip.unsqueeze(0), loss_funcs, args)
                    logger.output_video(ss_out[0][0].detach().cpu().numpy(), wandb_step, Fs, vid_name="resampled_"+metric_output_name)
                    saliency_maps = ss_out[1][0].permute(1,0,2,3).detach().cpu().numpy()
                    saliency_maps *= 1/np.max(saliency_maps)
                    logger.output_video(saliency_maps, wandb_step, Fs, vid_name="saliency_"+metric_output_name)

    # Aggregate and print epoch stats
    stats, all_hr = aggregate_epoch_stats(running_stats, all_hr, metric_output_name)

    # Log final test stats
    logger.output_epoch_stats(stats, all_hr, wandb_step, metric_output_name)
    return wandb_step + 10
