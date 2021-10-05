from enum import Enum, auto
from typing import Dict, List, Tuple

import numpy as np


ln = np.log
abs = np.abs
exp = np.exp
floor = np.floor
round = np.round


class AttrDict(dict):
    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)


def scheduler_wrapper(scheduler_func, step_size, steps_per_epoch, max_lr, base_lr, gamma, t_mult):
    scheduler_state = AttrDict(cycle=0, stepsize=step_size * steps_per_epoch, prev_cycle_end=0)

    def wrapped(iteration):
        remainder = iteration - scheduler_state.prev_cycle_end
        is_end_of_cycle = remainder >= scheduler_state.stepsize
        if is_end_of_cycle and iteration != 0:
            scheduler_state.cycle = scheduler_state.cycle + 1
            scheduler_state.prev_cycle_end += scheduler_state.stepsize
            scheduler_state.stepsize = steps_per_epoch * int(
                step_size * t_mult**scheduler_state.cycle)

        remainder = iteration - scheduler_state.prev_cycle_end
        x = remainder / (scheduler_state.stepsize - 1)
        lr = scheduler_func(x, scheduler_state.cycle, max_lr, base_lr, gamma)
        return lr, (remainder == scheduler_state.stepsize - 1)

    return wrapped


def simple(x: float, cycle: int, max_lr: float, base_lr: float, gamma=0.9):
    return max_lr


def jigsaw(x: float, cycle: int, max_lr: float, base_lr: float, gamma=0.9):
    max_lr = max_lr * (gamma**cycle)
    lr = base_lr + (max_lr - base_lr) * (1 - x)
    return lr


def jigsaw_log(x: float, cycle: int, max_lr: float, base_lr: float, gamma=0.9):
    if max_lr == 0.0:
        max_lr = 1e-12
    max_lr = max_lr * (gamma**cycle)
    lr = np.exp(np.log(base_lr) + (np.log(max_lr) - np.log(base_lr)) * (1 - x))
    lr = base_lr if x == 1 else lr
    return lr


def cosine(x: float, cycle: int, max_lr: float, base_lr: float, gamma=0.9):
    max_lr = max_lr * (gamma**cycle)
    lr = ((np.cos(np.pi * (x)) + 1) / 2) * (max_lr - base_lr) + base_lr
    return lr


class SchedulerType(Enum):
    Simple = auto()
    Jigsaw = auto()
    JigsawLog = auto()
    Cosine = auto()


cyclical_lr_funcs = {
    SchedulerType.Simple: simple,
    SchedulerType.Jigsaw: jigsaw,
    SchedulerType.JigsawLog: jigsaw_log,
    SchedulerType.Cosine: cosine
}


def list_based_scheduler_to_lr_array(lr_list: List[Tuple[int, float]], steps_per_epoch=1):
    lr_array = []
    for num_epochs, lr in lr_list:
        lr_array += [lr] * (steps_per_epoch * num_epochs)
    return lr_array


def lr_array_to_target_len(lr_array, target_len: int):
    lr_array = np.array(lr_array)
    if len(lr_array) >= target_len:
        return lr_array[:target_len]
    else:
        return np.concatenate([lr_array, [lr_array[-1]] * (target_len - len(lr_array))])


def generate_lr_array(scheduler_type: SchedulerType,
                      base_lr: float = 1e-6,
                      max_lr: float = 1e-2,
                      gamma: float = 0.3,
                      period_init_len=1,
                      period_exp_fac=1.,
                      steps_per_epoch: int = 1,
                      num_epochs: int = 1):

    if isinstance(scheduler_type, str):
        scheduler_type = SchedulerType[scheduler_type]
    elif isinstance(scheduler_type, SchedulerType):
        pass
    else:
        raise NotImplementedError()

    lr_func = scheduler_wrapper(cyclical_lr_funcs[scheduler_type],
                                step_size=period_init_len,
                                steps_per_epoch=steps_per_epoch,
                                max_lr=max_lr,
                                t_mult=period_exp_fac,
                                base_lr=base_lr,
                                gamma=gamma)
    return [lr_func(i)[0] for i in range(num_epochs * steps_per_epoch)]


def lr_array_to_sched_func(lr_array):
    def sched(epoch: int):
        return lr_array[epoch]

    return sched
