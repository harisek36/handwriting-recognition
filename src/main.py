import os
import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model
from SamplePreprocessor import preprocess
import glob
import numpy as np
import pathlib
import requests
import threading
import time
import os
from flask import jsonify
from flask_cors import CORS
import re
from flask import Flask, request
import sys
import base64
from werkzeug.utils import secure_filename

from datetime import datetime

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = '/Users/harishsekar/Documents/Project/DataScience/HTC_GLOBAL/SimpleHTR-master/src/imagesAngular'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = ''

class FilePaths:
    "filenames and paths to data"
    fnCharList = '/Users/harishsekar/Documents/Project/DataScience/HTC_GLOBAL/SimpleHTR-master/model/charList.txt'
    fnAccuracy = '/Users/harishsekar/Documents/Project/DataScience/HTC_GLOBAL/SimpleHTR-master/model/accuracy.txt'
    fnTrain = '/Users/harishsekar/Documents/Project/DataScience/HTC_GLOBAL/SimpleHTR-master/data'
    fnInfer= glob.glob ("/Users/harishsekar/Documents/Project/DataScience/HTC_GLOBAL/SimpleHTR-master/data/words/a01/a01-020/*.png")

def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 5 # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)
        
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        recognized = model.inferBatch(batch)
        print('Ground truth -> Recognized')    
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
    
    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate


def infer(model, fnImg):
    "recognize text in image provided by file path"

    f = open("testfile.doc","w") #opens file with name of "test.txt"
    
    for i in range(0, len(fnImg)):
        img = preprocess(cv2.imread(fnImg[i], cv2.IMREAD_GRAYSCALE), Model.imgSize)
        batch = Batch(None, [img] * Model.batchSize) # fill all batch elements with same input image
        recognized = model.inferBatch(batch) # recognize text
        f.write(recognized[0]+" ")
        print('Recognized:' , recognized[0])
        if (i%8)==0: f.newlines
    f.close()

           # all batch elements hold same result
    
def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the NN", action="store_true")
    parser.add_argument("--validate", help="validate the NN", action="store_true")
    parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
    args = parser.parse_args()

    # train or validate on IAM dataset    
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, args.beamsearch)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, args.beamsearch, mustRestore=True)
            validate(model, loader)

    # infer text on test image
    else:
        print(open('/Users/harishsekar/Documents/Project/DataScience/HTC_GLOBAL/SimpleHTR-master/model/accuracy.txt').read())
        model = Model(open(FilePaths.fnCharList).read(), args.beamsearch, mustRestore=True)
        infer(model, FilePaths.fnInfer)

@app.route('/handwritting', methods=['POST'])
def inferRestImage():

    print(open('/Users/harishsekar/Documents/Project/DataScience/HTC_GLOBAL/SimpleHTR-master/model/accuracy.txt').read())
    model = Model(open(FilePaths.fnCharList).read(), useBeamSearch=True, mustRestore=True)

    if 'imagefile' in request.files:
        imageFile = request.files['imagefile']
        filename = secure_filename(imageFile.filename)
        imageFile.save('/Users/harishsekar/Documents/Project/DataScience/HTC_GLOBAL/SimpleHTR-master/src/imagesAngular/'+filename)

        img = preprocess(cv2.imread('/Users/harishsekar/Documents/Project/DataScience/HTC_GLOBAL/SimpleHTR-master/src/imagesAngular/'+filename, cv2.IMREAD_GRAYSCALE), Model.imgSize)
        batch = Batch(None, [img] * Model.batchSize) # fill all batch elements with same input image
        recognized = model.inferBatch(batch) # recognize text
        return jsonify(text = recognized[0])

        # return jsonify(text = "Success")
    return jsonify(text = "Success")



@app.route("/start")
def initialSetup():
    return jsonify(status = 'initial setup started')

if __name__ == '__main__':
    app.run()



