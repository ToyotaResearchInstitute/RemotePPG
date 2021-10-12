import argparse
import os


def parse_args(mode):
    parser = argparse.ArgumentParser()

    if mode == 'train':
        # Model parameters
        model_parser = parser.add_argument_group('Model')
        model_parser.add_argument('model', type=str, help='PhysNet, MeanBaseline, MedianBaseline, FrequencyContrast')
        model_parser.add_argument("--pretrained_weights", type=str, default=None, help="if specified starts from checkpoint model")
        model_parser.add_argument("--model_channels", type=int, default=64, help="the number of channels to use in the model")

        # FrequencyContrast-specific parameters
        fc_parser = parser.add_argument_group('FrequencyContrast Model Parameters')
        fc_parser.add_argument('--contrast_model', type=str, default='PhysNet', help='the core model used for the FrequencyContrast model')
        fc_parser.add_argument('--mvtl_window_size', type=int, default=None, help='the subset window used for the FrequencyContrast model')
        fc_parser.add_argument('--mvtl_number_views', type=int, default=4, help='the number of views used for the FrequencyContrast model')
        fc_parser.add_argument('--mvtl_distance', type=str, default='PSD_MSE', help='the distance metric used for the FrequencyContrast model')

        # How to output results - wandb / output dir
        output_parser = parser.add_argument_group('Experiment Output')
        output_parser.add_argument("--output_dir", type=str, help="all output will be saved in this directory")
        output_parser.add_argument('--wandb_project_name', type=str, default=None, help='which wandb project to log output to (default: do not use wandb)')
        output_parser.add_argument('--group_name', type=str, default=None, help='interpretable name for wandb group')
        output_parser.add_argument('--exp_name', type=str, default=None, help='interpretable name for wandb run')
        output_parser.add_argument('--exp_tags', nargs='+', type=str, default=None, help='wandb experiment tags')
        output_parser.add_argument('--save_test_videos', type=int, default=None, help='save every n-th test video')        

        # Optimization and loss functions
        opt_parser = parser.add_argument_group('Optimization and Losses')
        opt_parser.add_argument('--loss', nargs='+', type=str, default=[], help='L1, MSE, NegPC, NegSNR, MVTL, NegMCC, IPR')
        opt_parser.add_argument('--loss_weights', nargs='+', type=float, default=None, help='The relative weight for each loss')
        opt_parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
        opt_parser.add_argument('--optimizer', type=str, default='adamw', help='Select optimizer: adamw, sgd')
        opt_parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
        opt_parser.add_argument('--scheduler', type=str, default=None, help='Select scheduler: step, exponential, cosine, cyclic')
        opt_parser.add_argument('--scheduler_params', nargs='+', type=float, default=None, help='Scheduler-specific parameters')
        opt_parser.add_argument("--exclude_border", type=int, default=0, help="the number of samples to exclude from metric calculation on each side")
        opt_parser.add_argument("--train_sampler_number", type=int, default=None, help="the number of random samples to draw per train epoch")

        # Validation
        val_parser = parser.add_argument_group('Validation')
        val_parser.add_argument('--val_on_train', default=False, action='store_true', help='skip the validation phase and instead run metric on train')
        val_parser.add_argument('--val_on_test', default=False, action='store_true', help='run metric on test set')
        val_parser.add_argument('--val_metric', nargs='+', type=str, default=[], help='The metric or list of metrics used to determine validation performance. Irrespective, the model is always also saved at the final epoch by default (model_Last).')

        # Parameter overrides used during the test phase
        test_parser = parser.add_argument_group('Test Parameter Overrides')
        test_parser.add_argument('--test_time_depth_extractor', type=int, help='test time depth for the extractor model (if different)')
        test_parser.add_argument('--test_chunk_dataset', default=False, action='store_true', help='chunk the test dataset into lengths of time_depth')
        test_parser.add_argument('--test_time_depth', type=int, default=None, help='temporal length of samples during test time')
        test_parser.add_argument('--test_filter_activities', nargs='+', default=None, help='select which activities to include at test time')
        test_parser.add_argument('--test_datasets', nargs='+', type=str, help='names of datasets to use for testing')
        test_parser.add_argument('--test_protocol', type=str, default=None, help='the method to divide folds (preset, preset_add_test, preset_mix_train_dev, round_robin, round_robin_no_dev, temporal, all)')
        test_parser.add_argument('--test_batch_size', type=int, default=None, help='the batch size to use during testing')
        test_parser.add_argument("--test_downsample", type=float, default=None, help="Percent of test data to keep")
        test_parser.add_argument("--test_exclude_border", type=int, default=None, help="the number of samples to exclude from metric calculation on each side")

        # Saliency sampler module
        ss_parser = parser.add_argument_group('Saliency Sampler')
        ss_parser.add_argument('--ss', type=int, default=0, help='toggle saliency sampler [0,1]')
        ss_parser.add_argument('--ss_pretrain', type=int, default=0, help='toggle saliency sampler ImageNet pretraining')
        ss_parser.add_argument('--ss_dim', default=None, type=int, nargs='+', help='width and height of saliency network input in pixels')
        ss_parser.add_argument('--ss_out_dim', default=None, type=int, nargs='+', help='width and height of saliency network input in pixels')
        ss_parser.add_argument('--ss_layers', default=None, type=int, help='number of layers to include in saliency net')
        ss_parser.add_argument('--ss_sparsity', default=0, type=float, help='weighting for sparsity loss')
        ss_parser.add_argument('--ss_temporal', default=0, type=float, help='weighting for temporal consistency loss')

        # Noise injection experiment
        noise_injection_parser = parser.add_argument_group('Noise Injection')
        noise_injection_parser.add_argument('--noise_block', default=None, type=int, help='block location [1:top_left]')
        noise_injection_parser.add_argument('--noise_size', default=10, type=int, help='block size (pixels)')
        noise_injection_parser.add_argument('--noise_period', nargs='+', type=int, default=[10, 30], help='period range in number of frames')
        noise_injection_parser.add_argument('--noise_duty', default=0.5, type=float, help='duty cycle')
    elif mode == 'features':
        # Only used for feature extraction
        feature_extraction_parser = parser.add_argument_group('Feature Extraction')
        feature_extraction_parser.add_argument('--crop_overwrite', default=False, action='store_true', help='overwrite any existing cached cropped face tracking')


    # USED FOR ALL SCRIPTS
    # Datasets selected
    dataset_selection_parser = parser.add_argument_group('Dataset Selection')
    dataset_selection_parser.add_argument('--datasets', nargs='+', type=str, help='names of datasets to use')
    dataset_selection_parser.add_argument('--unprocessed', default=False, action='store_true', help='flag to use raw instead of preprocessed data')

    # Protocol parameters
    protocol_parser = parser.add_argument_group('Train/Test Protocol')
    protocol_parser.add_argument('--protocol', type=str, default='preset', help='the method to divide folds (preset, preset_add_test, preset_mix_train_dev, round_robin, round_robin_no_dev, temporal, all)')
    protocol_parser.add_argument('--round_robin_index', type=int, default=0, help='the test index to use for round robin folds')
    protocol_parser.add_argument('--temporal_before_time', type=float, default=15.0, help='when using temporal split, training data is before temporal_before_time')
    protocol_parser.add_argument('--temporal_after_time', type=float, default=15.0, help='when using temporal split, test data is after temporal_after_time')

    # Feature selection
    feature_parser = parser.add_argument_group('Feature Selection')
    feature_parser.add_argument("--video_source", type=str, default="main", help="which video to use for each sample")
    feature_parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    feature_parser.add_argument('--time_depth', type=int, default=64, help='temporal length of samples')
    feature_parser.add_argument('--use_channels', nargs='+', type=str, default=['red', 'green', 'blue'], help='list of video channels to use (red, green, blue, bright, dark, subtract)')
    feature_parser.add_argument('--crop_aspect_ratio', type=float, default=1., help='crop input video to certain aspect ratio (ignored if crop height / width provided)')
    feature_parser.add_argument('--crop_height', type=int, default=None, help='crop the input video to a certain height')
    feature_parser.add_argument('--crop_width', type=int, default=None, help='crop the input video to a certain width')
    feature_parser.add_argument('--scale_height', type=int, default=128, help='scale the cropped video to a certain height')
    feature_parser.add_argument('--scale_width', type=int, default=128, help='scale the cropped video to a certain width')
    feature_parser.add_argument('--skip_videos', default=False, action='store_true', help='flag to not load videos')

    # Filtering samples
    filter_parser = parser.add_argument_group('Sample Filtering')
    filter_parser.add_argument('--resample_bad_ppg', default=False, action='store_true', help='resample train/val fold PPG when low to total frequency power is above 0.6')
    filter_parser.add_argument("--downsample", type=float, default=None, help="Percent of data to keep")
    filter_parser.add_argument("--dev_downsample", type=float, default=None, help="Percent of dev data to keep")
    filter_parser.add_argument('--filter_activities', nargs='+', default=None, help='select which activities to include')
    filter_parser.add_argument('--filter_sessions', type=int, nargs='+', default=None, help='select which sessions to include')
    filter_parser.add_argument('--filter_presets', nargs='+', default=None, help='select which preset folds to include')
    filter_parser.add_argument('--dev_filter_sessions', type=int, nargs='+', default=None, help='select which sessions to include')

    # Sample augmentation
    augmentation_parser = parser.add_argument_group('Sample Augmentation')
    augmentation_parser.add_argument("--inject_ppg_shift", type=float, default=0.0, help="Inject uniform random PPG shifts into ground truth (seconds)")
    augmentation_parser.add_argument('--freq_augm_range', nargs='+', type=float, default=None, help='range to apply frequency augmentation (default no augmentation)')
    augmentation_parser.add_argument('--img_augm', default=False, action='store_true', help='augment with image transforms')

    # Band pass filter
    band_pass_parser = parser.add_argument_group('Band Pass')
    band_pass_parser.add_argument("--high_pass", type=float, default=40.0, help="the high pass frequency to use for filtering (BPM units)")
    band_pass_parser.add_argument("--low_pass", type=float, default=250.0, help="the low pass frequency to use for filtering (BPM units)")

    # Experiment initialization
    experiment_parser = parser.add_argument_group('Experiment')
    experiment_parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')
    experiment_parser.add_argument('--random_seed', type=int, default=0, help='random seed to use for experiment')
    experiment_parser.add_argument("--num_repeats", type=str, default=None, help="used to specify repeat runs in the experiment launcher")


    # Verify arguments
    args = parser.parse_args()
    allowed_protocols = set(['preset', 'preset_add_test', 'preset_mix_train_dev', 'round_robin', 'round_robin_no_dev', 'temporal', 'all'])

    # Checks specific to train mode
    args.time_depth_extractor = args.time_depth
    if mode == 'train':
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
        if args.loss_weights is None:
            args.loss_weights = [1.] * len(args.loss)
        if args.mvtl_window_size is None:
            args.mvtl_window_size = int(args.time_depth / 2)
        if args.test_time_depth is None:
            args.test_time_depth = args.time_depth
        if args.test_protocol is not None and args.test_protocol not in allowed_protocols:
            raise Exception("Unknown test protocol " + args.test_protocol)
    else:
        args.test_chunk_dataset = False
    if args.protocol not in allowed_protocols:
        raise Exception("Unknown test protocol " + args.protocol)
    if args.dev_downsample is None:
        args.dev_downsample = args.downsample
    if args.dev_filter_sessions is None:
        args.dev_filter_sessions = args.filter_sessions

    return args
