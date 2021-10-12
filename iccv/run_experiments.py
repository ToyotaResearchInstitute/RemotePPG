#!/usr/bin/env python3

import argparse
import subprocess
import sys
import threading
import os
import shlex

import torch


class GPU():
    def __init__(self, id, manager):
        self.id = id
        self.manager = manager

    def release(self):
        self.manager.release_gpu(self)

    def __str__(self):
        return str(self.id)


class GPUManager():
    """ Track available GPUs and provide on request
    """
    def __init__(self, available_gpus):
        self.semaphore = threading.BoundedSemaphore(len(available_gpus))
        self.gpu_list_lock = threading.Lock()
        self.available_gpus = list(available_gpus)

    def get_gpu(self):
        self.semaphore.acquire()
        with self.gpu_list_lock:
            gpu = self.available_gpus.pop()
        return GPU(gpu, self)

    def get_gpus(self, num_gpu=1):
        gpu_list = []
        for ii in range(num_gpu):
            gpu_list.append(self.get_gpu())
        return gpu_list

    def release_gpu(self, gpu):
        with self.gpu_list_lock:
            self.available_gpus.append(gpu.id)
            self.semaphore.release()


def run_command_with_gpus(command, gpu_list):
    print(f'GPU {",".join([str(gpu) for gpu in gpu_list])}: {command}')

    def run_and_release(command, gpu_list):
        myenv = os.environ.copy()
        myenv['CUDA_VISIBLE_DEVICES'] = ",".join([str(gpu) for gpu in gpu_list])
        proc = subprocess.Popen(args=command,
                                shell=True,
                                env=myenv)
        proc.wait()
        for gpu in gpu_list:
            gpu.release()

    thread = threading.Thread(target=run_and_release,
                              args=(command, gpu_list))
    thread.start()
    return thread


def run_command_list(manager, command_list, num_gpu):
    for command in command_list:
        gpu_list = manager.get_gpus(num_gpu=num_gpu)
        run_command_with_gpus(command, gpu_list)


def read_commands(exp_file):
    with open(exp_file, 'r') as f:
        command_list = [line.rstrip() for line in f]
    return command_list


def expand_repeats(command_list, start_number):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_repeats', type=int, default=None)
    parser.add_argument('--protocol', type=str, default='preset')
    parser.add_argument('--test_protocol', type=str, default=None)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--ground_truth_weights', type=str, default=None)

    out_commands = []
    for command in command_list:
        args, _ = parser.parse_known_args(shlex.split(command))
        if args.num_repeats is None:
            out_commands.append(command)
        else:
            if args.test_protocol is None:
                args.test_protocol = args.protocol

            for repeat_it in range(args.num_repeats):
                new_command = command
                exp_num = repeat_it + start_number
                new_command += " --group_name " + args.exp_name
                new_command += " --exp_name " + args.exp_name + "-" + str(exp_num)
                if "round_robin" in args.test_protocol:
                    new_command += " --random_seed " + str(int(exp_num / 5))
                    new_command += " --round_robin_index " + str(exp_num % 5)
                else:
                    new_command += " --random_seed " + str(exp_num)
                if args.ground_truth_weights is not None:
                    new_command += " --ground_truth_weights " + args.ground_truth_weights.replace('{iteration}', str(exp_num))
                out_commands.append(new_command)
    return out_commands


def main():
    parser = argparse.ArgumentParser(description='Schedule a list of GPU experiments.')
    parser.add_argument('-e', '--exp_txt', type=str, required=True,
                        help='txt file with one line per command, see e.g. exp/example.txt')
    parser.add_argument('-g', '--gpus', nargs='+', type=str, default=[], required=False, help='which GPUs to use. If unset, will use all')
    parser.add_argument('-s', '--start_number', type=int, default=0, help='starting number for random seed and experiment name for multi-run experiments (default 0)')
    parser.add_argument('-n', '--num_gpu', type=int, default=1, help='number of GPUs to use per experiment (default 1)')
    args = parser.parse_args()

    gpus = args.gpus
    if len(gpus) == 0:
        # find all available gpus
        gpus = [str(x) for x in range(torch.cuda.device_count())]

    manager = GPUManager(gpus)
    exp_file = args.exp_txt
    command_list = read_commands(exp_file)
    command_list = expand_repeats(command_list, args.start_number)
    run_command_list(manager, command_list, args.num_gpu)


if __name__ == '__main__':
    main()

