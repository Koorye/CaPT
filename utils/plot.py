import argparse
import re
import matplotlib.pyplot as plt
import os
import os.path as osp
import pandas as pd
import seaborn as sns
from plotnine import *
from tqdm import tqdm


_INTEGER_PATTERN = r'^-?\d+$'
_FLOAT_PATTERN = r'^\-?\d+\.\d+$'


def parse_value(var_and_values, variables):
    outs = dict()
    
    for var_and_value in var_and_values.split('_'):
        for var in variables:
            if var in var_and_value:
                value = var_and_value.split(var)[-1]

                if re.match(_INTEGER_PATTERN, value):
                    value = int(value)
                elif re.match(_FLOAT_PATTERN, value):
                    value = float(value)
                
                outs[var] = value

    for var in variables:
        if var not in outs:
            return None

    return outs


def read_acc(path):
    df = pd.read_csv(path, index_col=None)
    df = df[['dataset', 'base_acc', 'new_acc', 'H']]
    return pd.melt(df, id_vars=['dataset'], value_vars=['base_acc', 'new_acc', 'H'],
                   var_name='acc_type', value_name='acc')


def read_accs(dir_, prefix, variables):
    filenames = [filename for filename in os.listdir(dir_) 
                 if filename.endswith('.csv') and filename.startswith(prefix)]
    paths = [osp.join(dir_, filename) for filename in filenames]
    dfs = [read_acc(path) for path in paths]

    dfs_new = []
    
    for df, filename in zip(dfs, filenames):
        filename = '.'.join(filename.split('.')[:-1])
        s = filename.split('-')
        trainer, cfg, var_and_values = s[1], s[2], ('-').join(s[3:])
        var_and_values = parse_value(var_and_values, variables)
        
        if var_and_values is None:
            continue

        df['trainer'] = trainer
        df['cfg'] = cfg
        for var, value in var_and_values.items():
            df[var] = value

        dfs_new.append(df)
        
    return pd.concat(dfs_new).reset_index(drop=True)


def plot_1d(df, variables, save_path, plot, average_only):
    if average_only:
        df = df[df['dataset'] == 'average']
        
    if len(variables) == 1:
        var = variables[0]
    else:
        var = '&'.join(variables)
        df[var] = df.apply(lambda row: '&'.join([str(row[var]) for var in variables]))
    
    if plot == 'line':
        (
            ggplot(df, aes(var, 'acc', color='acc_type', label='acc', group='acc_type'))
            + geom_line()
            + geom_point()
            + facet_wrap('dataset', scales='free_y')
            + theme_seaborn()
            + theme(axis_text_x=element_text(rotation=45))
        ).save(save_path, height=8, width=10)

    elif plot == 'col':
        (
            ggplot(df, aes(var, 'acc', fill='acc_type', label='acc', group='acc_type'))
            + geom_col(position='dodge')
            + facet_wrap('dataset', scales='free_y')
            + theme_seaborn()
            + theme(axis_text_x=element_text(rotation=45))
        ).save(save_path, height=8, width=10)


def plot_2d(df, variables, save_path, average_only):
    assert len(variables) == 2, 'length of variables should be 2!'
    var1, var2 = variables
    
    if average_only:
        df = df[df['dataset'] == 'average']

    datasets = df['dataset'].unique().tolist()

    plt.figure(figsize=(3 * 10, len(datasets) * 10))

    for r, dataset in enumerate(tqdm(datasets)):
        df_per_dataset = df[df['dataset'] == dataset]
        
        for c, acc_type in enumerate(['base_acc', 'new_acc', 'H']):
            df_per_acc_type = df_per_dataset[df_per_dataset['acc_type'] == acc_type][[var1, var2, 'acc']]
            df_2d = df_per_acc_type.set_index([var1, var2]).unstack()

            xticks = [idx[1] for idx in df_2d.columns]
            yticks = df_2d.index.tolist()

            plt.subplot(len(datasets), 3, r * 3 + c + 1)
            sns.heatmap(df_2d, annot=True, fmt='.2f', cbar=False,
                        xticklabels=xticks, yticklabels=yticks)
            plt.xlabel(var2)
            plt.ylabel(var1)
            plt.title(dataset + ' ' + acc_type)
    
    plt.savefig(save_path)
    plt.close()


def main(args):
    df = read_accs(args.dir, args.prefix, args.vars)
    if args.plot == 'heat':
        plot_2d(df, args.vars, args.save_path, args.average_only)
    else:
        plot_1d(df, args.vars, args.save_path, args.plot, args.average_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--vars', nargs='+', required=True)
    parser.add_argument('--plot', type=str, default='line')
    parser.add_argument('--average-only', action='store_true')
    args = parser.parse_args()
    main(args)
