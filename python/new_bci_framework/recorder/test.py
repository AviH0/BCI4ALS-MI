from time import sleep

from new_bci_framework.recorder.opeb_bci_cyton_recorder import *

def main_test():
    rec = CytonRecorder(None, BoardIds.SYNTHETIC_BOARD)
    rec.start_recording()
    sleep(5)
    rec.end_recording()
    raw = rec.get_raw_data()

    raw.plot()