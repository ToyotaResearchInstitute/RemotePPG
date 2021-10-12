import numpy as np
from collections import OrderedDict

import torch


def get_model_output(model, model_name, video, loss_funcs, args):
    batch_stats = {}

    # Get output signal
    #TODO set p to 0 during training (augmentation)
    ss_output = None
    if args.ss:
        model_output, ss_output = model(x=video, p=1)
    else:
        model_output = model(video)
    
    # Compute model-specific signals
    if model_name == 'FrequencyContrast':
        signals, branches = model_output
        if 'MVTL' in loss_funcs:
            batch_stats['ppg_MVTL'] = torch.mean(loss_funcs['MVTL'](branches))
    else:
        signals = model_output.squeeze(4).squeeze(3).squeeze(1)

    # Remove border
    if args.exclude_border != 0:
        signals = signals[:,args.exclude_border:-args.exclude_border]

    return signals, batch_stats, ss_output


def calculate_ppg_metrics(signal_preds, targets, ppg_metrics, loss_funcs):
    batch_stats = {}
    for ppg_metric in ppg_metrics:
        # Determine which target to use / whether to skip
        if ppg_metric == 'MVTL':
            continue
        if ppg_metric == 'NegSNR':
            signal_targets = targets['our']
        else:
            signal_targets = targets['ppg']
        
        # Calculate metric
        batch_stats['ppg_' + ppg_metric] = torch.mean(loss_funcs[ppg_metric](signal_preds, signal_targets))
    return batch_stats


def aggregate_epoch_stats(running_stats, epoch_hr, phase):
    # Log ppg epoch statistics
    num_entries = len(epoch_hr['our'])
    stats = OrderedDict()
    for key, value in running_stats.items():
        if 'inst' in key:
            stats[key] = value
        else:
            stats[key] = value / num_entries

    # Convert hr stats to numpy array
    for key, stat in epoch_hr.items():
        epoch_hr[key] = np.array(stat)

    # Calculate HR error stats
    for comparison_pair in [('our', 'gt'), ('pred', 'gt'), ('pred', 'our')]:
        comparison = comparison_pair[0] + '_v_' + comparison_pair[1]
        error_hr = epoch_hr[comparison_pair[1]] - epoch_hr[comparison_pair[0]]
        stats[f'{comparison}_sd'] = np.std(error_hr)
        stats[f'{comparison}_rmse'] = np.sqrt(np.mean(np.square(error_hr)))
        stats[f'{comparison}_mae'] = np.mean(np.abs(error_hr))
        stats[f'{comparison}_corr'] = np.corrcoef(epoch_hr[comparison_pair[1]], epoch_hr[comparison_pair[0]])[0,1]

    # Print all stats to the command line
    print()
    for key, value in stats.items():
        print('{} {}: {:.4f} '.format(phase, key, value))
    return stats, epoch_hr
