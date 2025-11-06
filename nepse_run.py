import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import matplotlib.pyplot as plt

def create_timesnet_args(**overrides):
    # Create Namespace with all default values from the parser
    args = argparse.Namespace(
        # Basic config
        task_name='long_term_forecast',
        is_training=1,
        model_id='nepse',
        model='TimesNet',

        # Data loader
        data='NEPSE',
        root_path='./dataset/nepse/',
        data_path='nepse.csv',
        # forecasting task, options:[M, S, MS];
        # M:multivariate predict multivariate,
        # S:univariate predict univariate,
        # MS:multivariate predict univariate'
        features='M',
        target='Close',
        # freq for time features encoding,
        # options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly],
        # you can also use more detailed freq like 15min or 3h')
        freq='d',
        checkpoints='./checkpoints/',

        # Forecasting task
        # Input Sequence Length
        seq_len=30,
        # Start token Length
        label_len=15,
        # Prediction Label Length
        pred_len=7,
        seasonal_patterns='Monthly',
        inverse=False,

        # Imputation task
        mask_rate=0.25,

        # Anomaly detection task
        anomaly_ratio=0.25,

        # Model define
        # expansion factor for Mamba
        expand=2,
        # conv kernel size for Mamba
        d_conv=4,
        # TimesBlock
        top_k=5,
        # For Inception
        num_kernels=6,
        # Encoder Input size
        enc_in=6,
        # Decoder Input Size
        dec_in=6,
        # Output size
        c_out=6,
        # Dimension of model
        d_model=128,
        # No of heads
        n_heads=8,
        # No of encoder layer
        e_layers=2,
        # No of decoder layer
        d_layers=2,
        # Dimension of Fully Connected Network
        d_ff=256,
        # Window size of Moving average in Time series
        moving_avg=21,
        # Attention Factor
        factor=6,
        # whether to use distilling in encoder, using this argument means not using distilling
        distil=True,
        dropout=0.1,
        embed='timeF',
        # Activation
        activation='gelu',
        # 0: channel dependence 1: channel independence for FreTS model
        channel_independence=1,
        # method of series decompsition, only support moving_avg or dft_decomp
        decomp_method='moving_avg',
        # whether to use normalize; True 1 False 0
        use_norm=1,
        # Num of down sampling layers
        down_sampling_layers=0,
        # Down sampling window size
        down_sampling_window=1,
        # Down sampling method only avg, max, conv
        down_sampling_method=None,
        # Length of segment-wise iteration of SegRNN
        seg_len=96,

        ## Optimization
        # Data loader num workers
        num_workers=2,
        # Experiments Times
        itr=1,
        # Train Epochs
        train_epochs=50,
        # Batch size of train data input
        batch_size=64,
        # early Stopping Patience
        patience=50,
        # Learning Rate
        learning_rate=0.0001,
        des='test',
        # Loss Function
        loss='MSE',
        # Adjust Learning Rate
        lradj='type1',
        # use automatic mixed Precision training
        use_amp=False,

        ## GPU
        use_gpu=True,
        gpu=0,
        gpu_type='cuda',
        use_multi_gpu=False,
        devices='0,1',

        ## De-stationary projector params
        # hidden layer dimensions of projector (List)
        p_hidden_dims=[128, 128],
        # No of hidden layers in projector
        p_hidden_layers=2,

        # Metrics (dtw)
        use_dtw=False,

        ## Augmentation
        # How many times to agumetn
        augmentation_ratio=0,
        # Randomization seed
        seed=2,
        # Jitter Preset Augmentation
        jitter=False,
        # Scaling Preset Augmentation
        scaling=False,
        # Equal Length Permutation preset augmentation
        permutation=False,
        # Random Length Permutation preset augmentation
        randompermutation=False,
        # Magnitude warp preset augmentation
        magwarp=False,
        # Time Warp preset augmenation
        timewarp=False,
        # Window slice preset augmentation
        windowslice=False,
        # Window Wrap Preset augmentation
        windowwarp=False,
        # Rotation Preset Augmentation
        rotation=False,
        # Spawner Preset Augmentation
        spawner=False,
        # DTW warp preset augmentation
        dtwwarp=False,
        # Shape DWT Wrap
        shapedtwwarp=False,
        # Weighted DAB preset augmentation
        wdba=False,
        # Discrimitive DTW warp preset augmentation
        discdtw=False,
        # Discrimitive shapeDTW warp preset augmentation
        discsdtw=False,
        # Anything extra
        extra_tag="",

        # TimeXer Patch Length
        patch_len=16
    )

    # Override with any provided arguments
    for key, value in overrides.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: Unknown argument '{key}' ignored")

    # Set device (same logic as original code)
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            print('Using MPS')
        else:
            args.device = torch.device("cpu")
            print('Using CPU')

    # Handle multi-GPU setup
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print_args(args)
    return args


def run_experiment(**arg_overrides):
    # Create args with overrides
    args = create_timesnet_args(**arg_overrides)

    # Import your experiment class
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    Exp = Exp_Long_Term_Forecast

    # Run experiments
    for ii in range(args.itr):
        # Initialize experiment
        exp = Exp(args)
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        # Training
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)
        # Testing
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        metrics, gt, preds = exp.test(setting)
        # Clear cache
        if args.gpu_type == 'mps':
            torch.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
    print("\nExperiment completed successfully!")
    return metrics, gt, preds


def plot_graph(true, preds=None):
    print("Plotting results...")
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    metrics, gt, preds = run_experiment(
        train_epochs=2,
        batch_size=64,
        learning_rate=0.001,
        pred_len=14,
        seq_len=60
    )
    plot_graph(gt, preds)

