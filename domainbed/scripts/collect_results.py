# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from .. import datasets
from .. import algorithms
from ..lib import misc, reporting
from .. import model_selection
from ..lib.query import Q
import warnings

import matplotlib.pyplot as plt
def plot_traces(traces, show_s=True):
    # 绘制每个 trace 的折线图
    for i, (name, trace) in enumerate(traces):
        # 获取每个数据点的 acc_s 和 acc_t 值，并按照 step 排序
        data = sorted([(d['step'], d['acc_s'], d['acc_t']) for d in trace])
        steps, acc_s, acc_t = zip(*data)

        # 绘制 acc_t 的折线图，并添加算法名称到图例
        plt.plot(steps, acc_t, linestyle='-', color=f'C{i}', label=f'{name}')

        # 如果需要显示 acc_s，则绘制 acc_s 的折线图
        if show_s:
            plt.plot(steps, acc_s, linestyle='--', color=f'C{i}', label=f'{name}')

    plt.ylim([0, 1])

    # 设置图例的位置和样式
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

    # 添加标签和标题
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Step')

    # 显示图形
    plt.show()



def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, selection_method, latex, show_traces):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")
    

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    print(len(alg_names))
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            if len(means) > 0:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))
            else:
                table[i][-1] = '?'

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)
    
    if show_traces:
        traces  =[]
        for i, algorithm in enumerate(alg_names):
            for j, dataset in enumerate(dataset_names):
                trace = []
                for r in records:
                    if r['args']['algorithm'] == algorithm and r['args']['dataset'] == dataset:
                        r: dict
                        def get_env_id(key: str):
                            id_str = key[key.find('v')+1:]
                            id_str = id_str[:id_str.find('_')]
                            return int(id_str)
                        acc = {get_env_id(key): r[key] for key in r.keys() if 'env' in key and 'in_acc' in key}
                        acc_s = [acc[key] for key in acc.keys() if key not in r['args']['test_envs']]
                        acc_t = [acc[key] for key in acc.keys() if key in r['args']['test_envs']]
                        trace.append({'step': r['step'], 'acc_s': sum(acc_s)/len(acc_s), 'acc_t': sum(acc_t)/len(acc_t)})
            
                traces.append((f'{algorithm}', trace))
        plot_traces(traces, False)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True, nargs='+')
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir[0], results_file), "w")

    print(args.input_dir)
    records = reporting.load_records_multiple_paths(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        model_selection.LeaveOneOutSelectionMethod,
        model_selection.InformationHeatSelectionMethod,
        model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args.latex, args.traces)

    if args.latex:
        print("\\end{document}")
