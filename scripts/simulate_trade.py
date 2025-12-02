import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from utils import io_tools
from utils.trade import trade
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_theme(style='whitegrid', context='paper', font_scale=2)
palette = sns.color_palette('muted')

ROOT = io_tools.get_root(__file__, num_returns=2)

LABEL_DICT = {
    'cmamba': 'CryptoMamba',
    'lstm': 'LSTM',
    'lstm_bi': 'Bi-LSTM',
    'gru': 'GRU',
   'smamba': 'S-Mamba',
    'itransformer': 'iTransformer',
}

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu',
        help="The type of accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of computing devices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Logging directory.",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default='Cmamba',
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )

    parser.add_argument(
        "--balance",
        type=float,
        default=100,
        help="initial money",
    )

    parser.add_argument(
        "--risk",
        type=float,
        default=2,
    )

    parser.add_argument(
        "--split",
        type=str,
        default='test',
        choices={'test', 'val', 'train'},
    )

    parser.add_argument(
        "--trade_mode",
        type=str,
        default='smart',
        choices={'smart', 'smart_w_short', 'vanilla', 'no_strategy', 'smart_prob'},
    )

    args = parser.parse_args()
    return args

def load_model(config, ckpt_path, config_name=None):
    if ckpt_path is None:
        ckpt_path = f'{ROOT}/checkpoints/{config_name}.ckpt'
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get('normalize', False)
    model_class = io_tools.get_obj_from_str(model_config.get('target'))
    model = model_class.load_from_checkpoint(ckpt_path, **model_config.get('params'))
    model.cuda()
    model.eval()
    return model, normalize


def init_dirs(args, name):
    path = f'{ROOT}/Results/{name}/{args.config}'
    if name == 'all':
        path = f'{ROOT}/Results/all/'
    if not os.path.isdir(path):
        os.makedirs(path)

def max_drawdown(prices):
    prices = np.array(prices)
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    mdd = drawdown.min()
    return -mdd


@torch.no_grad()
def run_model(model, dataloader, factors=None, is_classification=False):
    target_list = []
    preds_list = []
    timetamps = []
    probs_list = [] # To store full probability distributions
    
    with torch.no_grad():
        for batch in dataloader:
            ts = batch.get('Timestamp').numpy().reshape(-1)
            features = batch.get('features').to(model.device)
            
            # Run model
            output = model(features)
            
            # Handle Classification vs Regression
            if is_classification:
                # output shape: (B, num_classes)
                # Apply Softmax to get probabilities
                probs = torch.softmax(output, dim=1).cpu().numpy()
                
                # For single prediction, we can take argmax (class) or expected value
                # Let's store the class prediction
                preds = np.argmax(probs, axis=1)
                
                # Get targets (class indices)
                target = batch.get('target_class').numpy().reshape(-1)
                
                preds_list += [int(x) for x in list(preds)]
                probs_list += [x for x in list(probs)]
                
            else:
                # Regression output
                preds = output.cpu().numpy().reshape(-1)
                target = batch.get(model.y_key).numpy().reshape(-1)
                preds_list += [float(x) for x in list(preds)]
            
            target_list += [float(x) for x in list(target)]
            
            if factors is not None:
                timetamps += [float(x) for x in list(batch.get('Timestamp_orig').numpy().reshape(-1))]
            else:
                timetamps += [float(x) for x in list(ts)]

    if not is_classification and factors is not None:
        scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
        shift = factors.get(model.y_key).get('min')
        target_list = [x * scale + shift for x in target_list]
        preds_list = [x * scale + shift for x in preds_list]

    targets = np.asarray(target_list)
    preds = np.asarray(preds_list)
    probs = np.asarray(probs_list) if is_classification else None

    return timetamps, targets, preds, probs


if __name__ == '__main__':
    args = get_args()
    init_dir_flag = False
    colors = ['darkblue', 'yellowgreen', 'crimson', 'darkviolet', 'orange', 'magenta']
    if args.config == 'all':
        config_list = [x.replace('.ckpt', '') for x in os.listdir(f'{ROOT}/checkpoints/') if '_nv.ckpt' in x]
    elif args.config == 'all_v':
        config_list = [x.replace('.ckpt', '') for x in os.listdir(f'{ROOT}/checkpoints/') if '_v.ckpt' in x]
        init_dirs(args, 'all')
    else:
        config_list = [args.config]
        colors = ['darkblue']
        init_dir_flag = True
    
    plt.figure(figsize=(15, 10))
    for conf, c in zip(config_list, colors):
        config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{conf}.yaml')
        if init_dir_flag:
            init_dir_flag = False
            init_dirs(args, config.get('name', args.expname))
        data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
        
        model, normalize = load_model(config, args.ckpt_path, config_name=conf)
        
        # Check if model is classification based on config
        num_classes = config.get('params', {}).get('num_classes', None)
        is_classification = num_classes is not None

        use_volume = config.get('use_volume', False)
        test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
        data_module = CMambaDataModule(data_config,
                                        train_transform=test_transform,
                                        val_transform=test_transform,
                                        test_transform=test_transform,
                                        batch_size=args.batch_size,
                                        distributed_sampler=False,
                                        num_workers=args.num_workers,
                                        normalize=normalize,
                                        window_size=config.get('params', {}).get('window_size', 14)
                                        )
        
        if args.split == 'test':
            test_loader = data_module.test_dataloader()
        if args.split == 'val':
            test_loader = data_module.val_dataloader()
        if args.split == 'train':
            test_loader = data_module.train_dataloader()

        factors = None
        if normalize:
            factors = data_module.factors
            
        timstamps, targets, preds, probs = run_model(model, test_loader, factors, is_classification)
        
        # Process predictions for trading
        trading_preds = preds
        
        # For classification, calculate expected return
        expected_returns = None
        if is_classification:
            # Bin definitions (should match config/dataset)
            min_r = data_config.get('binning', {}).get('min_range', -3.5)
            max_r = data_config.get('binning', {}).get('max_range', 3.5)
            step = data_config.get('binning', {}).get('step', 0.25)
            
            # Create bin centers
            # Bin 0: < min_r (we'll assume min_r - step)
            # Bin N: > max_r (we'll assume max_r + step)
            # Middle bins: min_r + step/2, etc.
            num_bins = int((max_r - min_r) / step) + 2 # +2 for underflow/overflow
            
            bin_values = []
            # Underflow bin
            bin_values.append(min_r - step)
            # Middle bins
            current = min_r
            while current < max_r:
                bin_values.append(current + step/2)
                current += step
            # Overflow bin
            bin_values.append(max_r + step)
            
            bin_values = np.array(bin_values)
            
            # Calculate expected return: sum(prob * bin_value)
            expected_returns = np.sum(probs * bin_values, axis=1)
            trading_preds = expected_returns

        # Concatenate all data splits to ensure historical data availability for trading simulation
        full_data = pd.concat(data_module.data_dict.values())
        # Remove duplicates just in case, though splits should be distinct
        full_data = full_data.drop_duplicates(subset=['Timestamp']).sort_values(by=['Timestamp']).reset_index(drop=True)
        
        data = full_data
        tmp = data.get('Close')
        time_key = 'Timestamp'
        if normalize:
            time_key = 'Timestamp_orig'
            # Only de-normalize data if we are in regression mode where preds are prices
            # In classification mode, preds are classes/returns, but we still need prices for trading simulation
            scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
            shift = factors.get(model.y_key).get('min')
            data[model.y_key] = data[model.y_key] * scale + shift

        balance, balance_in_time = trade(data, time_key, timstamps, targets, trading_preds, 
                                         balance=args.balance, mode=args.trade_mode, 
                                         risk=args.risk, y_key=model.y_key, is_classification=is_classification)
        
        # Save predictions to CSV
        output_data = {
            'Timestamp': timstamps,
            'Date': [datetime.fromtimestamp(int(x)) for x in timstamps],
            'Actual': targets, # Class index or price
        }
        
        if is_classification:
            output_data['Predicted_Class'] = preds
            output_data['Expected_Return'] = expected_returns
            # Optionally add max prob
            output_data['Max_Prob'] = np.max(probs, axis=1)
        else:
            output_data['Predicted_Price'] = preds
            
        output_df = pd.DataFrame(output_data)
        
        # Create filename based on config and split
        csv_filename = f"{ROOT}/Results/{config.get('name', args.expname)}/{conf}_predictions_{args.split}.csv"
        output_df.to_csv(csv_filename, index=False)
        print(f"Predictions saved to: {csv_filename}")

        print(f'{conf} -- Final balance: {round(balance, 2)}')
        print(f'{conf} -- Maximum Draw Down : {round(max_drawdown(balance_in_time) * 100, 2)}')

        label = conf.replace("_nv", "").replace("_v", "")
        label = LABEL_DICT.get(label)
        tmp = [timstamps[0] - 24 * 60 * 60] + timstamps
        tmp = [datetime.fromtimestamp(int(x)) for x in tmp]
        sns.lineplot(x=tmp, 
                     y=balance_in_time, 
                     color=c, 
                     zorder=0, 
                     linewidth=2.5, 
                     label=label)

    name = config.get('name', args.expname)
    if args.trade_mode == 'no_strategy':
        plot_path = f'./balance_{args.split}.jpg'
    else:
        if len(config_list) == 1:
            plot_path = f'{ROOT}/Results/{name}/{args.config}/balance_{args.split}_{args.trade_mode}.jpg'
        else:
            plot_path = f'{ROOT}/Results/all/balance_{args.config}_{args.split}_{args.trade_mode}.jpg'
    plt.xticks(rotation=30)
    plt.axhline(y=100, color='r', linestyle='--')

    if len(config_list) == 1:
        ax = plt.gca()
        ax.get_legend().remove() 
        plt.title(f'Balance in time (final: {round(balance, 2)})')
    else:
        plt.title(f'Net Worth in Time')

    # matplotlib.rcParams.update({'font.size': 100})
    plt.xlim([tmp[0], tmp[-1]])
    plt.ylabel('Balance ($)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')