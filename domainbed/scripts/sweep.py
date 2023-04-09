# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from .. import datasets
from .. import hparams_registry
from .. import algorithms
from ..lib import misc
from .. import command_launchers

import tqdm
import shlex

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python', '-m', 'codes.models.baselines.DomainBed.domainbed.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        for i, job in enumerate(jobs):
            commands = [job.command_str]
            launcher_fn(commands)
            print(f'Launched {i}/{len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, steps,
    data_dir, task, holdout_fraction, single_test_envs, hparams, device, test_envs, featurizer):
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                if single_test_envs:
                    if test_envs and len(test_envs) > 0:
                        all_test_envs = [[i] for i in test_envs]
                    else:
                        all_test_envs = [
                            [i] for i in range(datasets.num_environments(dataset))]
                else:
                    if test_envs and len(test_envs) > 0:
                        raise NotImplemented()
                    all_test_envs = all_test_env_combinations(
                        datasets.num_environments(dataset))
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['holdout_fraction'] = holdout_fraction
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['trial_seed'] = trial_seed
                        train_args['device'] = device
                        train_args['featurizer'] = featurizer
                        train_args['seed'] = misc.seed_hash(dataset,
                            algorithm, test_envs, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--single_test_envs', action='store_true')
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--test_envs', nargs='+', type=int, default=[])
    parser.add_argument('--featurizer', type=str, default='CNN')
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams,
        device=args.device,
        test_envs=args.test_envs,
        featurizer=args.featurizer
    )

    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
