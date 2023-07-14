# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np
from collections import deque
from math import sqrt

def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        # chosen_record = run_records.sorted(lambda r: r[test_in_acc_key])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }

class OracleAllSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        chosen_record = run_records.sorted(lambda r: r[test_in_acc_key])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }


class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None

from .lib.query import Q
class InformationHeatSelectionMethod(SelectionMethod):
    name="selection by distortion and inductive bias"
    beta = 0.1
    window_size = 5

    @classmethod
    def run_acc(self, run_records: Q):
        if run_records[0]['args']['steps'] < 5000:
            return None
        min_Q_F = run_records.map(lambda rec: rec['Q_F']).min()
        if min_Q_F < 0:
            return None
        train_domain = run_records[0]['args']['train_envs']
        test_domain = run_records[0]['args']['test_envs']
        assert len(train_domain) == 1 and len(test_domain) == 1
        train_domain = train_domain[0]
        test_domain = test_domain[0]
        train_acc = f'env{train_domain}_in_acc'
        test_acc = f'env{test_domain}_uda_acc' # see UDA transductive

        run_records = run_records.sorted(key=lambda x: x['step'])
        variance = 0
        window = deque([0] + [float(row[train_acc]) for row in run_records[:self.window_size-1]])
        for i in range(self.window_size, len(run_records)):
            window.popleft()
            window.append(float(run_records[i][train_acc]))
            mean = sum(window) / len(window)
            variance = sqrt(sum([(a - mean)**2 for a in window]) / (len(window) - 1))
            run_records[i]['variance'] = variance
            run_records[i]['min_in_window'] = min(window)
        
        for i in range(self.window_size):
            run_records[i]['variance'] = 1 # 舍弃前几个 record
            run_records[i]['min_in_window'] = 0
        fluctuation = 0
        run_records[0]['fluctuation'] = 1
        for i in range(1, len(run_records)):
            fluctuation = self.beta * fluctuation + (1 - self.beta) * abs(run_records[i][train_acc] - run_records[i-1][train_acc])
            run_records[i]['fluctuation'] = fluctuation

        selected: dict = (
            run_records
            .filter(lambda x: x[train_acc] >= 0.95)
            # .filter(lambda x: x['variance'] <= 0.05)
            # .filter(lambda x: x['fluctuation'] <= 0.05)
            .filter(lambda x: x['min_in_window'] >= 0.90)
            # .filter(lambda x: x['Q_F'] + x['Q_0'] >= 5)
            .sorted(key=lambda x: x['n_inductive_bias_difference'])[-1]
        )
        distortions = selected['distortions']
        return {
            'val_acc': - sum(distortions) / len(distortions) if len(distortions) > 0 else 0,
            'test_acc': selected[test_acc],
            'etc': {
                'difference': selected['n_inductive_bias_difference'],
                'heat': selected['Q_F'] + selected['Q_0'],
                'step': selected['step']
            }
        }
    @classmethod
    def sweep_acc(self, records):
        if records[0]['args']['algorithm'] != 'InformationalHeat':
            return None
        else:
            return super().sweep_acc(records)

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        res = (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

        print(res[0][1][0]['args']['hparams_seed'], res[0][0])
        return res
