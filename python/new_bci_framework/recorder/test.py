from time import sleep

from new_bci_framework.recorder.opeb_bci_cyton_recorder import *
from new_bci_framework.recorder.plot_rt_recording import Graph


def main_test():
    rec = CytonRecorder(None, BoardIds.SYNTHETIC_BOARD)
    rec.start_recording()
    rec.plot_live_data()
    sleep(15)
    rec.end_recording()
    raw = rec.get_raw_data()

    raw.plot()