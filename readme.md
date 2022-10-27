# NATURAL LANGUAGE INFERENCE DETECTION WITH UNIVERSAL SENTENCE ENCODER


Tensorflow/Keras implementation of language inference detection using Universal sentence encoders and sentence embeddings.

Data :<br/>
----
Stanford Natural Language Inference Corpus
A collection of 570k labeled human-written English sentence pairs : https://www.kaggle.com/datasets/stanfordu/stanford-natural-language-inference-corpus

File Description :
----
- model.py : model written with tensoflow/keras
- train.py : training file
- prediction.py : prediction file for deployment purpose
- requirements.txt : required packages and versions to run model

Usage :
if necessary download repo and create an virtual env using following commands 
----
download file 
```
conda create --name exp_env
conda activate exp_env
```
find the folder directory in exp_env
```
pip install -r requirements.txt 
```
run ***train.py*** file. 
<br/>
NOTE:
If necessary one can run the train.py ans save the model weights. After that, model weights can be used for prediction. Since, size of the model_weights file, it will not be shared in github.
<br/>
for deployment purpose prediction file created seperately as **prediction.py**


dataset url:
https://www.kaggle.com/datasets/stanfordu/stanford-natural-language-inference-corpus

Evaluation:
----
```
F1 SCORE : 0.7910290031945397


precision    recall  f1-score   support

contradiction       0.89      0.76      0.82      3333
   entailment       0.75      0.87      0.80      3333
      neutral       0.75      0.75      0.75      3334

     accuracy                           0.79     10000
    macro avg       0.80      0.79      0.79     10000
 weighted avg       0.80      0.79      0.79     10000
```
