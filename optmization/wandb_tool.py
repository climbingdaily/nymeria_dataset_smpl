################################################################################
# File: /wanb_tool.py                                                          #
# Created Date: Wednesday April 12th 2023                                      #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2023 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

import os

import numpy as np
import wandb
import torch
from yacs.config import CfgNode


def save_wandb(raw_loss, init_loss_chart, loss_chart, start, lenght, person='first_person'):
    """
    It takes the loss dictionaries and logs them to wandb

    Args:
      raw_loss: the loss of the first person before optimization
      init_loss_chart: the loss chart of the initial model
      loss_chart: the loss chart for the optimized model
      start: the frame number to start the optimization from
      lenght: the number of frames to be optimized
      person: The person you want to optimize. Defaults to first_person
    """
    for i in range(0, lenght):
        loss_summary = {'frames': i + start}

        def set_lossi(loss, start=0, strs='init'):
            for k in loss:
                if type(loss[k][1]) != list:
                    ld = loss[k][1].tolist()
                else:
                    ld = loss[k][1]
                if i + start in ld:
                    loss_summary[f'{strs}_{k}'] = loss[k][0][ld.index(
                        i + start)]

        set_lossi(raw_loss, start, 'init')
        set_lossi(init_loss_chart, 0, 'int')
        set_lossi(loss_chart, 0, 'optmized')

        wandb.log({person: loss_summary})


def plot_lines(x, ys, title, names=None, person='first.person'):
    yss = [[j.item() for j in y] for y in ys]
    wandb.log({f'{person}_{title}': wandb.plot.line_series(
        x, yss, keys=names, title=f'{person}_{title}', xname='frames')})


def set_wandb(args):
    if not args.wandb or args.debug:
        os.environ['WANDB_MODE'] = 'disabled'
        # args.opt_end = args.opt_start + 800
        # args.window_frames = 100
    else:
        args.name = args.name + '_wandb'
    if args.offline:
        os.environ['WANDB_MODE'] = 'offline'

    cfg = CfgNode.load_cfg(
        open(os.path.join(os.path.dirname(__file__), 'config.yaml')))
    wandb.init(project='SLOPER4D', entity='climbingdaily', resume='allow',
               group=os.path.basename(args.root_folder))
    wandb.run.name = f'{os.path.basename(args.root_folder)}-{args.name}-{wandb.run.id}'

    wandb.config.update(args, allow_val_change=True)
    wandb.config.update(cfg, allow_val_change=True)

    config = wandb.config
    return config


def set_loss_dict(loss_list, weight, loss_dict, indexes, num_list=1, prefix='sliding', concat=False):
    """
    This function sets the loss dictionary for a given prefix and index based on the provided loss list,
    weight, and other parameters.
    
    Args:
      loss_list: A list of losses to be aggregated and computed.
      weight: The weight parameter is a scalar value that is multiplied with the computed loss to give
    the weighted loss. It is used to adjust the relative importance of different losses in a multi-loss
    scenario.
      loss_dict: A dictionary that stores the loss values for different prefixes and indexes.
      indexes: The indexes parameter is a tuple containing two values - the current index and the total
    number of iterations. These values are used to keep track of the progress of the training process.
      num_list: The parameter `num_list` is an optional integer parameter with a default value of 1. It
    is used as a multiplier for the `loss` value calculated in the function. Defaults to 1
      prefix: The prefix parameter is a string that is used to identify the type of loss being
    calculated. It can take on values such as 'sliding', 'contact', 'rot', 'pose', or 'm2p'. Defaults to
    sliding
      concat: A boolean flag indicating whether to concatenate the losses or not. If set to True, the
    function will concatenate the losses, otherwise it will compute the average loss. Defaults to False
    
    Returns:
      two values: `weight_loss` and `str`.
    """

    weight_loss, str = 0, ''

    scale = 180 / np.pi if prefix == 'rot' or prefix == 'pose' else 100

    index, iters = indexes

    if prefix not in loss_dict:
        loss_dict[prefix] = {}
    if index not in loss_dict[prefix]:
        loss_dict[prefix][index] = {}

    target = loss_dict[prefix][index]

    if loss_list is not None and len(loss_list) > 0:
        loss, num = 0, 0
        for l in loss_list:
            if l:
                if prefix in ['sliding', 'contact', 'm2p']:
                    # if prefix in ['m2p']:
                    l = l * l.item() * 100
                # if l > 0.03 and prefix in ['trans']:
                #     l = l * l.item() * 100
                loss += l
                num += 1

        if num > 0:
            loss /= num

            print_loss = torch.as_tensor(loss_list)[torch.as_tensor(
                loss_list) > 0].mean().item() * scale

            if concat:
                if 'concat_loss' not in target:
                    target['concat_loss_no_opt'] = print_loss

                target['concat_loss'] = print_loss
                str = f'c{prefix} {print_loss:.3f} '
            else:
                if 'loss' not in target:
                    target['num'] = num
                    target['loss_no_opt'] = print_loss
                target['loss'] = print_loss
                str = f'{prefix} {print_loss:.3f} '

            weight_loss = weight * loss

    return weight_loss, str


def get_weight_loss(loss_list, weight, category, print_str, loss_dict, indexes, loss_chart, is_concat=False):
    """
    This function adds a loss item to a loss chart and returns the weight loss.
    
    Args:
      loss_list: A tuple containing the loss tensor and a list of numbers.
      weight: The weight of the loss item, which is used to calculate the overall loss of a model during
    training.
      category: The category of the loss item, which is used to organize and group the loss items in the
    loss chart.
      print_str: A list of strings to be printed later.
      loss_dict: A dictionary that stores the loss values for different categories.
      indexes: It is a variable that contains the indexes of the current batch being processed. These
    indexes are used to identify the specific samples in the batch that contribute to the loss
    calculation.
      is_concat: is_concat is a boolean parameter that determines whether the loss values should be
    concatenated with existing values in the loss chart for a given category or not. If is_concat is
    True, the new loss values will be added to the existing values in the loss chart. If is_concat is
    False, the existing values. Defaults to False
      loss_chart: a list that stores the loss values and corresponding numbers for each category
    
    Returns:
      the value of `weight_loss`.
    """

    loss, num_list = loss_list
    weight_loss, str = set_loss_dict(
        loss, weight, loss_dict, indexes, num_list, category, is_concat)

    if weight_loss and weight:
        if is_concat:
            loss_chart[category] = [[i.item() for i in loss] + loss_chart[category][0],
                                    num_list + loss_chart[category][1]]
        else:
            loss_chart[category] = [[i.item() for i in loss], num_list]

        print_str.append(str)
    else:
        weight_loss = 0

    return weight_loss
