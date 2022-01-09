from ..recorder.recorder import Recorder
from ..classifier.base_classifier import BaseClassifier
from ..paradigm.paradigm import Paradigm
from ..preprocessing.preprocessing_pipeline import PreprocessingPipeline
from ..config.config import Config


class Session:
    """
    Base class for an EEG session, with online or offline recording, or analysis of previous recordings.
    simple public api for creating and running the session.
    """

    def __init__(self, recorder: Recorder, paradigm: Paradigm, preprocessor: PreprocessingPipeline,
                 classifier: BaseClassifier, config: Config):
        self.recorder = recorder
        self.paradigm = paradigm
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.config = config
        self.raw_data = None
        self.epoched_data = None

    def run_all(self):
        pass

    @staticmethod
    def load_session(session_dir: str):
        """
        Load a previously recorded session from disk to preform analysis.
        :param session_dir: saved session directory
        :return: Session object
        """
