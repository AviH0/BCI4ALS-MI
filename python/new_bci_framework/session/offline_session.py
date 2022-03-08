from session import Session
from ..recorder.recorder import Recorder
from ..classifier.base_classifier import BaseClassifier
from ..paradigm.paradigm import Paradigm
from ..preprocessing.preprocessing_pipeline import PreprocessingPipeline
from ..config.config import Config


class OfflineSession(Session):
    """
    Subclass of session for an offline recording session.
    """

    def __init__(self, recorder: Recorder, paradigm: Paradigm, preprocessor: PreprocessingPipeline,
                 classifier: BaseClassifier, config: Config):
        super().__init__(recorder, paradigm, preprocessor, classifier, config)

    def run_recording(self):
        self.recorder.start_recording()
        self.paradigm.start(self.recorder)
        self.recorder.end_recording()

    def run_preprocessing(self):
        self.raw_data = self.recorder.get_raw_data()
        self.epoched_data = self.preprocessor.run_pipeline(self.raw_data)

    def run_classifier(self):
        train_data, test_data = train_test_split(self.epoched_data)
        self.classifier.fit(train_data)
        evaluation = self.classifier.evaluate(test_data)

    def run_all(self):
        self.run_recording()
        self.run_preprocessing()
        self.run_classifier()
