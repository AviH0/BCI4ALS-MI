import datetime

import mne


class Config:
    """
    class containing config information for a session.
    This should include any configurable parameters of all the other classes, such as
    directory names for saved data and figures, numbers of trials, train-test-split ratio, etc.
    """

    SUBJECT_NAME = ""
    DATE = datetime.datetime.now().date().isoformat()


class Recorder:
    """
    Take care of all aspects of recording the eeg data. If any other program need to be opened manually
     then this class should prompt the user to do that.
     public API for starting and stopping a recording, for pushing markers if a recording is in progress and
      for retrieving the data from the last recording as mne.Raw.
    """

    def __init__(self, config: Config):
        pass

    def start_recording(self):
        pass

    def push_marker(self):
        pass

    def end_recording(self):
        pass

    def get_raw_data(self) -> mne.io.Raw:
        pass


class PreprocessingPipeline:
    """
    A Preprocessing pipeline. In essence it receives a raw data object and returns teh segmented data as
    an Epochs object, after preforming filters, cleaning etc.
    Further design of this class can allow subclassing or other forms of modularity, allowing us to easily
    swap different pipelines.
    """

    def __init__(self, config: Config):
        pass

    def run_pipeline(self, data: mne.io.Raw) -> mne.Epochs:
        pass


class Paradigm:
    """
    This class decides the experiment paradigm. It holds all the information regarding the
    types of stimulus, the number of classes, trials etc.
    All the things that need to happen during the recording for this paradigm are under this classes
    responsibility (showing a gui, creating events, pushing markers to an active recording, etc.)
    Sublcasses may represent different paradigms such as p300, MI, etc.
    """

    def __init__(self, config: Config):
        pass

    def start(self, recorder: Recorder):
        pass


class P300Paradaigm(Paradigm):
    """
    Paradigm subclass for the p300 paradigm.
    """

    def __init__(self, config: Config):
        super(P300Paradaigm, self).__init__(config)


class BaseClassifier:
    """
    Basic class for a classifier for session eeg data.
    API includes training, prediction and evaluation.
    """

    def __init__(self, config: Config):
        pass

    def fit(self, data=mne.Epochs):
        pass

    def predict(self, data: mne.Epochs):
        pass

    def evaluate(self, data: mne.Epochs):
        pass


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
        train_data, test_data = train_test_split(epoched_data)
        self.classifier.fit(train_data)
        evaluation = self.classifier.evaluate(test_data)

    def run_all(self):
        self.run_recording()
        self.run_preprocessing()
        self.run_classifier()


if __name__ == '__main__':
    config = Config()

    config.SUBJECT_NAME = "TEST_SUBJECT"

    session = Session(
        recorder=Recorder(config),
        paradigm=P300Paradaigm(config),
        preprocessor=PreprocessingPipeline(config),
        classifier=BaseClassifier(config),
        config=config
    )
    session.run_all()
