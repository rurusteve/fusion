import argparse
import json
import os
import time
import pickle


import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import accuracy_score 

from config import CONFIG_BY_KEY
from text_data_loader import TextDataLoader
from text_data_loader import TextDataHelper

from audio_data_loader import AudioDataLoader
from audio_data_loader import AudioDataHelper

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

RESULT_FILE = "./output/{}.json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-key', default='', choices=list(CONFIG_BY_KEY.keys()))
    return parser.parse_args()


args = parse_args()
print("Args:", args)

# Load config
config = CONFIG_BY_KEY[args.config_key]

# Load textdata
textdata = TextDataLoader(config)
audiodata = AudioDataLoader(config)


def latefusion(y_pred, y_true):
    if config.use_context:
            print("Use Context")
    if config.use_author:
            print("Use Author")

    if config.use_target_text:
        if config.use_bert:
            print("BERT Model")
        else:
            print("GLOVE Model")
    else:
        print("Audio Only")
    cf_matrix = confusion_matrix(y_true, y_pred)
    result_string = classification_report(y_true, y_pred, digits=3)
    print(y_true)
    print(y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print (accuracy_score(y_true, y_pred))
    sns.heatmap(cf_matrix, annot=True)

    print("Accuracy:")
    print(result_string)
    print("Runtime:")
    print("--- %s seconds ---" % (time.time() - start_time))

    return classification_report(y_true, y_pred, output_dict=True, digits=3), result_string

def svm_train_text(text_train_input, text_train_output):
    print("SVM Train Text")
    textclf = make_pipeline(
        StandardScaler() if config.svm_scale else FunctionTransformer(lambda x: x, validate=False),
        svm.SVC(C=config.svm_c, gamma='scale', kernel='rbf', probability=True)
    )
    
    return textclf.fit(text_train_input, np.argmax(text_train_output, axis=1))


def svm_test_text(textclf, text_test_input, text_test_output):
    print("SVM Test Text")
    probas = textclf.predict(text_test_input)
    y_pred = probas
    y_true = np.argmax(text_test_output, axis=1)


    latefusion(y_pred, y_true)

def svm_train_audio(audio_train_input, audio_train_output):
    print("SVM Train Audio")
    audioclf = make_pipeline(
        StandardScaler() if config.svm_scale else FunctionTransformer(lambda x: x, validate=False),
        svm.SVC(C=config.svm_c, gamma='scale', kernel='rbf', probability=True)
    )

    return audioclf.fit(audio_train_input, np.argmax(audio_train_output, axis=1))


def svm_test_audio(audioclf, audio_test_input, audio_test_output):
    print("SVM Test Audio")
    probas = audioclf.predict(audio_test_input)
    y_pred = probas
    y_true = np.argmax(audio_test_output, axis=1)

    latefusion(y_pred, y_true)

def svm_train(audio_train_input, audio_train_output, text_train_input, text_train_output):
    print("SVM Train Bimodal")
    audioclf = make_pipeline(
        StandardScaler() if config.svm_scale else FunctionTransformer(lambda x: x, validate=False),
        svm.SVC(C=config.svm_c, gamma='scale', kernel='rbf', probability=True)
    )

    textclf = make_pipeline(
        StandardScaler() if config.svm_scale else FunctionTransformer(lambda x: x, validate=False),
        svm.SVC(C=config.svm_c, gamma='scale', kernel='rbf', probability=True)
    )

    return audioclf.fit(audio_train_input, np.argmax(audio_train_output, axis=1)), audioclf.fit(audio_train_input, np.argmax(audio_train_output, axis=1))


def svm_test(audioclf, audio_test_input, audio_test_output, textclf, text_test_input, text_test_output):

    #print(audioclf)
    
    #print(text_test_input)
    print(len(text_test_input[0]))
    
    #print(text_test_output)
    print(len(text_test_output[0]))


    print("SVM Test Bimodal")
    if config.use_target_text:
        textprobas = textclf.predict(text_test_input)
        y_pred_text = textprobas
        y_true_text = np.argmax(text_test_output, axis=1)

    if config.use_target_audio:
        audioprobas = audioclf.predict(audio_test_input)
        y_pred_audio = audioprobas
        y_true_audio = np.argmax(audio_test_output, axis=1)

    if (config.use_target_text) and (config.use_target_audio):
        textprobas = textclf.predict(text_test_input)
        audioprobas = audioclf.predict(audio_test_input)
        txtprobas = textclf.predict(text_test_input)
        y_pred = np.concatenate([textprobas, audioprobas])
        
        y_true_text = np.argmax(text_test_output, axis=1)
        y_true_audio = np.argmax(audio_test_output, axis=1)
        y_true = np.concatenate([y_true_text, y_true_audio]) 
    else:
        if config.use_target_text:
            y_pred = y_pred_text
            y_true = y_true_text
        if config.use_target_audio:
            y_pred = y_pred_audio
            y_true = y_true_audio

    #print("Text Probas")
    #print(textprobas)
    #print("Audio Probas")
    #print(audioprobas)
    
    #realprobas = list(map(lambda x, y:x + y, textprobas, audioprobas)) 
    #print(realprobas)

    print("Y Pred")
    print(audioprobas)
    print(textprobas)
    print("Text Test Output")
    print(text_test_output)
    print("Text clf")
    print(textclf)

    #array1 = audioprobas
    #array2 = textprobas
    #result = [[1, 1] if (max(x+y) in y)&(max(x+y) in x) else([1,0] if max(x+y) in y else [0, 1]) for x,y in zip(array1,array2)]
    #print(result)

    #realprobas = np.concatenate([textprobas, audioprobas])
    #print(realprobas)

    #print("Text Pred")
    #print(y_pred_text)
    #print("Pred")
    #print(y_pred)
    
    #print("Concatenated Text & Audio Output")
    #print(y_true)
    
    latefusion(y_pred, y_true)

def texttrainIO(text_train_index, text_test_index):

   
    if config.use_target_text:

        # Prepare textdata
        text_train_input, text_train_output = textdata.getSplit(text_train_index)
        text_test_input, text_test_output = textdata.getSplit(text_test_index)

        datahelper = TextDataHelper(text_train_input, text_train_output, text_test_input, text_test_output, config, textdata)

        text_train_input = np.empty((len(text_train_input), 0))
        text_test_input = np.empty((len(text_test_input), 0))

        if config.use_bert:
            text_train_input = np.concatenate([text_train_input, datahelper.getTargetBertFeatures(mode='train')], axis=1)
            text_test_input = np.concatenate([text_test_input, datahelper.getTargetBertFeatures(mode='test')], axis=1)
        else:
            text_train_input = np.concatenate([text_train_input,
                                          np.array([datahelper.pool_text(utt)
                                                    for utt in datahelper.vectorizeUtterance(mode='train')])], axis=1)
            text_test_input = np.concatenate([text_test_input,
                                         np.array([datahelper.pool_text(utt)
                                                   for utt in datahelper.vectorizeUtterance(mode='test')])], axis=1)
    
    if text_train_input.shape[1] == 0:
        print("Invalid modalities")
        exit(1)

    # Aux input

    if config.use_author:
        train_input_author = datahelper.getAuthor(mode="train")
        test_input_author =  datahelper.getAuthor(mode="test")

        text_train_input = np.concatenate([text_train_input, train_input_author], axis=1)
        text_test_input = np.concatenate([text_test_input, test_input_author], axis=1)

    if config.use_context:
        if config.use_bert:
            train_input_context = datahelper.getContextBertFeatures(mode="train")
            test_input_context =  datahelper.getContextBertFeatures(mode="test")
        else:
            train_input_context = datahelper.getContextPool(mode="train")
            test_input_context =  datahelper.getContextPool(mode="test")

        text_train_input = np.concatenate([text_train_input, train_input_context], axis=1)
        text_test_input = np.concatenate([text_test_input, test_input_context], axis=1)
   

    text_train_output = datahelper.oneHotOutput(mode="train", size=config.num_classes)
    text_test_output = datahelper.oneHotOutput(mode="test", size=config.num_classes)

    return text_train_input, text_train_output, text_test_input, text_test_output

def audiotrainIO(audio_train_index, audio_test_index):

    if config.use_target_audio:

        # Prepare audiodata
        audio_train_input, audio_train_output = audiodata.getSplit(audio_train_index)
        audio_test_input, audio_test_output = audiodata.getSplit(audio_test_index)

        datahelper = AudioDataHelper(audio_train_input, audio_train_output, audio_test_input, audio_test_output, config, audiodata)

        audio_train_input = np.empty((len(audio_train_input), 0))
        audio_test_input = np.empty((len(audio_test_input), 0))
        
        audio_train_input = np.concatenate([audio_train_input, datahelper.getTargetAudioPool(mode='train')], axis=1)
        audio_test_input = np.concatenate([audio_test_input, datahelper.getTargetAudioPool(mode='test')], axis=1)

    if audio_train_input.shape[1] == 0:
        print("Invalid modalities")
        exit(1)
    
    audio_train_output = datahelper.oneHotOutput(mode="train", size=config.num_classes)
    audio_test_output = datahelper.oneHotOutput(mode="test", size=config.num_classes)

    return audio_train_input, audio_train_output, audio_test_input, audio_test_output

def trainSpeakerIndependent(model_name=None):

    config.fold = "SI"
    # if config.use_target_text:
    #     (text_train_index, text_test_index) = textdata.getSpeakerIndependent()
    #     text_train_input, text_train_output, text_test_input, text_test_output = texttrainIO(text_train_index, text_test_index)

    #     textclf = svm_train_text(text_train_input, text_train_output)
    #     svm_test_text(textclf, text_test_input, text_test_output)

    # if config.use_target_audio:
    #     (audio_train_index, audio_test_index) = audiodata.getSpeakerIndependent()
    #     audio_train_input, audio_train_output, audio_test_input, audio_test_output = audiotrainIO(audio_train_index, audio_test_index)

    #     audioclf = svm_train_audio(audio_train_input, audio_train_output)
    #     svm_test_audio(audioclf, audio_test_input, audio_test_output)

    if (config.use_target_text) and (config.use_target_audio):
        (text_train_index, text_test_index) = textdata.getSpeakerIndependent()
        text_train_input, text_train_output, text_test_input, text_test_output = texttrainIO(text_train_index, text_test_index)

        (audio_train_index, audio_test_index) = audiodata.getSpeakerIndependent()
        audio_train_input, audio_train_output, audio_test_input, audio_test_output = audiotrainIO(audio_train_index, audio_test_index)

        textclf = svm_train_text(text_train_input, text_train_output)
        audioclf = svm_train_audio(audio_train_input, audio_train_output)

        svm_test(audioclf, audio_test_input, audio_test_output, textclf, text_test_input, text_test_output)
    else:
        if config.use_target_text:
            (text_train_index, text_test_index) = textdata.getSpeakerIndependent()
            text_train_input, text_train_output, text_test_input, text_test_output = texttrainIO(text_train_index, text_test_index)
            textclf = svm_train_text(text_train_input, text_train_output)

            svm_test_text(textclf, text_test_input, text_test_output)
  
        if config.use_target_audio:
            (audio_train_index, audio_test_index) = audiodata.getSpeakerIndependent()
            audio_train_input, audio_train_output, audio_test_input, audio_test_output = audiotrainIO(audio_train_index, audio_test_index)

            audioclf = svm_train_audio(audio_train_input, audio_train_output)

            svm_test_audio(audioclf, audio_test_input, audio_test_output)
   


def trainSpeakerDependent(model_name=None):
    
    # Load textdata
    textdata = TextDataLoader(config)

    # Iterating over each fold
    results=[]
    for fold, (text_train_index, text_test_index) in enumerate(textdata.getStratifiedKFold()):

        # Present fold
        config.fold = fold+1
        print("Present Fold: {}".format(config.fold))

        text_train_input, text_train_output, text_test_input, text_test_output = texttrainIO(text_train_index, text_test_index)
        audio_train_input, audio_train_output, audio_test_input, audio_test_output = audiotrainIO(audio_train_index, audio_test_index)

        textclf = svm_train_text(text_train_input, text_train_output)
        audioclf = svm_train_audio(audio_train_input, audio_train_output)

        result_dict, result_str = svm_test_text(audioclf, audio_test_input, audio_test_output, textclf, text_test_input, text_test_output)

        results.append(result_dict)

    # Dumping result to output
    if not os.path.exists(os.path.dirname(RESULT_FILE)):
        os.makedirs(os.path.dirname(RESULT_FILE))
    with open(RESULT_FILE.format(model_name), 'w') as file:
        json.dump(results, file)


def printResult(model_name=None):

    results = json.load(open(RESULT_FILE.format(model_name), "rb"))

    weighted_precision, weighted_recall = [], []
    weighted_fscores = []

    print("#"*20)
    for fold, result in enumerate(results):
        weighted_fscores.append(result["weighted avg"]["f1-score"])
        weighted_precision.append(result["weighted avg"]["precision"])
        weighted_recall.append(result["weighted avg"]["recall"])

        print("Fold {}:".format(fold+1))
        print("Weighted Precision: {}  Weighted Recall: {}  Weighted F score: {}".format(result["weighted avg"]["precision"],
                                                                                         result["weighted avg"]["recall"],
                                                                                         result["weighted avg"]["f1-score"]))
    print("#"*20)
    print("Avg :")
    print("Weighted Precision: {:.3f}  Weighted Recall: {:.3f}  Weighted F score: {:.3f}".format(np.mean(weighted_precision),
                                                                                                 np.mean(weighted_recall),
                                                                                                 np.mean(weighted_fscores)))
 

if __name__ == "__main__":

    start_time = time.time()
    


    if config.speaker_independent:
        trainSpeakerIndependent(model_name=config.model)
    else:
        for _ in range(config.runs):
            trainSpeakerDependent(model_name=config.model)
            printResult(model_name=config.model)