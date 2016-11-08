# ner-exercise
This is an implementation of using deep learning for the Named-Entity Recognizer. We have creted three representations over the dataset, namely: 

1. Character-Level CNN representation over words.
2. Word-embedding representation. 
3. Classic hand crafted featuers. 

After combining these features, we used Biderectional Long Short Memeory (BLSTM) to generate tags for each word. 

We used Linear Chain Conditional Randome Field as a baseline.

Check the <https://github.com/mjlaali/ner-exercise/blob/master/document.pdf> document for more detail.

## Train Model 
To train the models, clone the github repository and create a folder called 'data'. Then, put two files with the name of 'train.txt' and 'test.txt'. These two files contain IOB tag information of the sentences. Please also copy the 'wordvecs.txt' file in 'data' folder. Finally, run the 'cnn_lstm_train.py' or 'crf_train.py' to train the CNN-LSTM model or the CRF model, respectively.

## Test Model
To test the learned model, please run either 'test_cnn_lstm_model.py' or 'test_crf_model.py' scripts.  

# Dependencies 
1. Python 3.5
2. Keras <https://keras.io/>
3. Tensorflow <https://www.tensorflow.org/>
3. Python-crfsuite <https://python-crfsuite.readthedocs.io/en/latest/>

