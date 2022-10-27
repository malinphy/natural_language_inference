import numpy as np 
import pandas as pd 
import tensorflow_hub as hub
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.layers import *

from sklearn.metrics import confusion_matrix,f1_score,classification_report
from sklearn.preprocessing import LabelEncoder
from model import model

use  = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") ## universal sentence encoder model

# from model import model
import pickle

train_path = 'C:/Users/user2/Google Drive/Colab Notebooks/datasets/snli/snli_1.0_train.csv'
test_path = 'C:/Users/user2/Google Drive/Colab Notebooks/datasets/snli/snli_1.0_test.csv'
validation_path = 'C:/Users/user2/Google Drive/Colab Notebooks/datasets/snli/snli_1.0_dev.csv'

selected_columns = ['sentence1','sentence2','label1']

train_df = pd.read_csv(train_path,usecols = selected_columns)
train_df = train_df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis", "label1": "label"})
test_df =  pd.read_csv(test_path,usecols = selected_columns)
test_df = test_df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis", "label1": "label"})
validation_df =  pd.read_csv(validation_path,usecols = selected_columns)
validation_df = validation_df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis", "label1": "label"})

train_df = train_df.dropna().reset_index(drop = True)
test_df = test_df.dropna().reset_index(drop = True)
validation_df = validation_df.dropna().reset_index(drop = True)

LE = LabelEncoder()
LE.fit(train_df['label'])
train_df['label'] = enc_label = LE.transform(train_df['label'])
test_df['label'] = enc_label = LE.transform(test_df['label'])
validation_df['label'] = enc_label = LE.transform(validation_df['label'])


output = open('LE.pkl', 'wb')
pickle.dump(LE, output)
output.close()

pkl_file = open('LE.pkl', 'rb')
LE = pickle.load(pkl_file) 
pkl_file.close()



# conv_size = 10
conv_size = 40
drop_rate = 0.2
pool_size = 6


use_model = model()
use_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(
                                        # learning_rate = 0.006
    ),
    metrics= ['accuracy'])

# history=use_model.fit(
#     [train_df['premise'], train_df['hypothesis']],
#     train_df['label'],
#     epochs = 2,
#     batch_size = 20,
#     validation_split = 0.2
#                     )



use_model.load_weights('C:/Users/user2/Google Drive/Colab Notebooks/snli/USE_snli_weights.h5')
# C:/Users/user2/Google Drive/Colab Notebooks/


print('END OF THE SCRIPT')