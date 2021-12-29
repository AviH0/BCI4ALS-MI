import logging
from typing import Tuple

import pandas as pd
import skorch
import torch
from braindecode import EEGClassifier
from braindecode.datautil.mne import create_from_mne_epochs
from braindecode.models import EEGNetv4
from braindecode.models import ShallowFBCSPNet
from braindecode.models import Deep4Net
from braindecode.util import set_random_seeds
from matplotlib import pyplot as plt
import os.path
import scipy.io
import numpy as np
import mne
import pyxdf
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from python.ica_nn_layer import ICAEEGNet

from skorch.callbacks import Checkpoint
from torch import Tensor
from skorch.helper import predefined_split

RECORDINGS_DIR = 'C:\Recordings'

def read_matlab_mat_to_np(filename, data_index=3):
    """
    Read a nummpy array from a matlab matrix file (.mat)
    :param filename: Matlab Matrix Filename (.mat file)
    :param data_index: Top level index of the data to be fetched. default is 3.
    :return:
    """
    # try:
    mat = scipy.io.loadmat(filename)
    return np.array(mat[list(mat.keys())[data_index]])


def get_epoched_data(subject, recording_dir="", epoch_duration=5) -> Tuple[mne.io.Raw, mne.Epochs]:
    if not recording_dir:
        recording_dir = RECORDINGS_DIR

    fname = f"{recording_dir}/{subject}/EEG.xdf"
    streams, headers = pyxdf.load_xdf(fname)
    marker_stream = None
    data_stream = None
    for stream in streams:
        stream_type = stream['info']['type']
        if stream_type == ["Markers"]:
            marker_stream = stream
        elif stream_type == ["EEG"]:
            data_stream = stream

    assert data_stream and marker_stream

    data = data_stream['time_series']

    stim = np.zeros(len(data))
    trial_started = False
    trial_type = 0
    for timestamp, marker in zip(marker_stream['time_stamps'], marker_stream['time_series']):
        # marker = int(float(marker[0]))
        # if marker == 1111:
        #     trial_started = True
        # if marker in [1, 2, 3]:
        #     trial_type = marker
        # if marker == 9:
        #     trial_started = False
        # if trial_started:
        index = np.where(np.isclose(data_stream['time_stamps'], timestamp, rtol=0, atol=0.01))
        # assert np.array(index).any()
        # if len(index):
        #     index = index[0]
        stim[index] = marker

    data = np.hstack([data, stim.reshape(-1, 1)]).T

    montage = mne.channels.read_custom_montage(fname="montage_ultracortex.loc")
    info = mne.create_info(sfreq=125, ch_names=list(montage.get_positions()['ch_pos'].keys())+["stim"], ch_types=['eeg']*16 + ["stim"])
    info.set_montage(montage)
    raw = mne.io.RawArray(data, info)

    event_dict = {"R/Right": 3, "R/Left": 2, "IR/Idle": 1}
    events = mne.find_events(raw, stim_channel="stim", output="onset", shortest_event=0)

    data *= 1e-6
    raw = mne.io.RawArray(data, info)
    raw.drop_channels(["stim", "T8", "PO3", "PO4"])



    raw.filter(l_freq=1, h_freq=40)
    raw.notch_filter(50, picks=['eeg'])
    raw = mne.preprocessing.compute_current_source_density(raw)
    epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=epoch_duration,
                                 event_id=event_dict, verbose='INFO', on_missing='warn')

    return raw, epochs


def create_windows(epochs: mne.Epochs, windows=False):
    old_level = mne.set_log_level(logging.FATAL, return_old_level=True)

    if windows:
        window_size_samples = 125 * 1  # 1 second
        window_stride_samples = 125//2  # half a second,
    else:
        window_size_samples = epochs[0].get_data().shape[2]
        window_stride_samples = epochs[0].get_data().shape[2]
    epochs.event_id = {"Right": 0, "Left": 1}
    epochs.events[:, 2][epochs.events[:, 2] == 2] = 0
    windows_datasets = create_from_mne_epochs(
                [epochs],
                window_size_samples=window_size_samples,
                window_stride_samples=window_stride_samples,
                drop_last_window=True
            )
    mne.set_log_level(old_level)


    indices = np.arange(0, len(windows_datasets.description))
    train_indices, test_indices = train_test_split(indices, train_size=0.7)

    splitted = windows_datasets.split(by=[list(train_indices), list(test_indices)])
    train_indices = train_indices
    test_indices = test_indices
    train_set = splitted['0']
    valid_set = splitted['1']
    windows_dataset = windows_datasets

    return windows_dataset, train_set, valid_set

class Criterion(torch.nn.NLLLoss):

    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print("input", input, sep='\n')
        target[target == 2] = 0
        # print("target", target, sep='\n')
        return super(Criterion, self).forward(input, target)

def get_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_dir = f'{checkpoint_dir}/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    chkpnt = Checkpoint(
        monitor='valid_accuracy_best',
        f_params="params{last_epoch[epoch]}.pt",
        f_optimizer="optimizer{last_epoch[epoch]}.pt",
        f_criterion="criterion{last_epoch[epoch]}.pt",
        f_history="history.json",
        dirname=checkpoint_dir
    )
    return chkpnt

def create_classifier(train_set, valid_set, checkpoint_dir, mixing_mat=None, unmixing_mat=None, batch_size=4):
    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = 20211128  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = 2
    # Extract number of chans and time steps from dataset
    n_chans = train_set[0][0].shape[0]
    input_window_samples = train_set[0][0].shape[1]
    F1 = 2
    D = 2
    F2 = 1
    model = ICAEEGNet(ShallowFBCSPNet, mixing_mat, unmixing_mat,
    # model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
        # pool_time_length=2,
        # pool_time_stride=2,
        # n_filters_time=2,
        # n_filters_spat=2,
        # n_filters_3=4,
        # n_filters_4=8
        # ,
        # kernel_length=125 // 2,
        # F1=F1,
        # D=D,
        # F2=F2,
        # drop_prob=0.5
    )
    # Send model to GPU
    if cuda:
        model.cuda()
    __device = device
    model = model


    lr = 0.1 * 0.001 # From https://braindecode.org/auto_examples/plot_bcic_iv_2a_moabb_trial.html
    wd = 0.5 * 0.001

    batch_size = batch_size



    chkpnt = get_checkpoint(checkpoint_dir)
    callbacks = [
        "accuracy", ("checkpoint", chkpnt),
        ("progress", skorch.callbacks.ProgressBar(detect_notebook=True))]



    classifier = EEGClassifier(
                model,
                criterion=Criterion,
                optimizer=torch.optim.Adam,
                train_split=predefined_split(valid_set),  # using valid_set for validation
                optimizer__lr=lr,
                optimizer__weight_decay=wd,
                batch_size=batch_size,
                callbacks=callbacks,
                device=__device,
            )

    return classifier

def train_classifier(classifier, n_epochs, train_set):
    old_level = mne.set_log_level(logging.FATAL, return_old_level=True)
    y = [dataset.y for dataset in train_set.datasets]
    classifier.fit(train_set, y=None, epochs=n_epochs)
    mne.set_log_level(old_level)


def load_best_checkpoint(checkpoint_dir, classifier, windows_dataset, valid_set):
    old_level = mne.set_log_level(logging.FATAL, return_old_level=True)
    chkpnt = get_checkpoint(checkpoint_dir)
    classifier.load_params(checkpoint=chkpnt)
    print(
                f"Total Unbalanced Score on Dataset: "
                f"{classifier.score(windows_dataset, np.hstack([np.array(trial.y) for trial in windows_dataset.datasets]))}")
    print(
                f"Total Unbalanced Score on Validation: "
                f"{classifier.score(valid_set, np.hstack([np.array(trial.y) for trial in valid_set.datasets]))}")
    mne.set_log_level(old_level)


def plot_train_results(classifier, figures_dir):
    results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
    df = pd.DataFrame(classifier.history[:, results_columns], columns=results_columns,
                      index=classifier.history[:, 'epoch'])

    # get percent of misclass for better visual comparison to loss
    df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                   valid_misclass=100 - 100 * df.valid_accuracy)

    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(figsize=(20, 10))
    df.loc[:, ['train_loss', 'valid_loss']].plot(
        ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
    ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    df.loc[:, ['train_misclass', 'valid_misclass']].plot(
        ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
    ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
    ax1.set_xlabel("Epoch", fontsize=14)

    # where some data has already been plotted to ax
    handles = [Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'),
               Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid')]
    plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/training_results.png")
    plt.show(block=True)
