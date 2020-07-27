import argparse
import json
import os

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import CONFIG_BY_KEY
from data_loader import DataLoader
from data_loader import DataHelper

RESULT_FILE = "./output/{}.json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-key', default='', choices=list(CONFIG_BY_KEY.keys()))
    return parser.parse_args()

args = parse_args()
print("Args:", args)

# Load config
config = CONFIG_BY_KEY[args.config_key]

# Load data
data = DataLoader(config)

def svm_train(train_input, train_output):
    clf = make_pipeline(
        StandardScaler() if config.svm_scale else FunctionTransformer(lambda x: x, validate=False),
        svm.SVC(C=config.svm_c, gamma='scale', kernel='rbf')
    )

    return clf.fit(train_input, np.argmax(train_output, axis=1))



if __name__ == "__main__":

    if config.speaker_independent:
        trainSpeakerIndependent(model_name=config.model)
    else:
        for _ in range(config.runs):
            trainSpeakerDependent(model_name=config.model)
            printResult(model_name=config.model)
