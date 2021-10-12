#!/usr/bin/env python3

import os
import argparse

from src.preprocessing.PURE import Process_PURE
from src.preprocessing.COHFACE import Process_COHFACE
from src.preprocessing.MR_NIRP_Car import Process_MR_NIRP_Car
from src.preprocessing.UBFC import Process_UBFC


# Overwrite links, but not files / dirs
def overwrite_link(src_path, dest_path):
    if os.path.islink(dest_path):
        os.remove(dest_path)
    if not os.path.exists(dest_path):
        os.symlink(src_path, dest_path)

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='The name of the dataset to process')
    parser.add_argument('--in_dir', type=str, help='The input dataset directory (default - check in ../data/raw/NAME)')
    parser.add_argument('--out_dir', type=str, default=None, help='The path where to store the output processed data (default - store in ../data/preprocessed/NAME)')
    parser.add_argument('--protocols', type=str, default=None, help='The directory of any protocol files to include')
    args = parser.parse_args()

    # Create raw and preprocessed dirs
    data_raw_path = 'data/raw'
    os.makedirs(data_raw_path, exist_ok = True)
    data_preprocessed_path = 'data/preprocessed'
    os.makedirs(data_preprocessed_path, exist_ok = True)

    # Add soft link to raw data
    raw_link_path = os.path.join(data_raw_path, args.name)
    if args.in_dir is not None:
        overwrite_link(args.in_dir, raw_link_path)
    args.in_dir = raw_link_path

    # Create preprocessed dir
    processed_link_path = os.path.join(data_preprocessed_path, args.name)
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok = True)
        overwrite_link(args.out_dir, processed_link_path)
    else:
        os.makedirs(processed_link_path, exist_ok = True)
    args.out_dir = processed_link_path

    # Process dataset
    print(f'Processing {args.name} dataset...')
    if args.name == 'PURE':
        Process_PURE(args)
    elif args.name == 'COHFACE':
        Process_COHFACE(args)
    elif args.name == 'MR-NIRP-Car':
        Process_MR_NIRP_Car(args)
    elif args.name == 'UBFC':
        Process_UBFC(args)
    else:
        print(f'ERROR: Unknown dataset {args.name}')
        exit(1)
    print('Finished!')
