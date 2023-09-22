#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset  # Import the Dataset class
from transformers import ElectraTokenizer, ElectraForPreTraining, AdamW
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import keras
from tqdm import tqdm
import pickle
from keras.models import Model
import keras.backend as K
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import itertools
from keras.models import load_model
from sklearn.utils import shuffle
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig


# In[ ]:


import torch
import torch.nn as nn
from transformers import ElectraTokenizer, ElectraForMaskedLM, ElectraForPreTraining
import pandas as pd


# In[ ]:


from sklearn.utils import shuffle
df = pd.read_csv(r"C:\Users\Yashika\OneDrive\Desktop\Dissertation\data set\Final Dataset\Racial and ethical hate speech.csv")
df = df.sample(frac=1).reset_index(drop=True) #
df = df[['label','text']]
df = shuffle(df)  
df.tail()


# In[ ]:


df['label'] = df['label'].map({ 'nothate':0,'hate':1})


# In[ ]:


df = df[['label','text']]
df = shuffle(df) 


# In[ ]:


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    #w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w


# In[ ]:


df['text']=df['text'].map(preprocess_sentence)


# In[ ]:


print(df['label'])


# In[ ]:


import pandas as pd
from sklearn.utils import resample
minority_class = df['label'].value_counts().idxmin()
print(minority_class)
# Separate majority and minority classes
majority_df = df[df['label'] != minority_class]
minority_df = df[df['label'] == minority_class]


# In[ ]:


# Downsample the majority class
majority_downsampled = resample(majority_df, replace=False, n_samples=len(minority_df), random_state=100)

# Combine the minority class DataFrame with the downsampled majority class DataFrame
balanced_df = pd.concat([majority_downsampled, minority_df])

# 'balanced_df' now contains a balanced dataset


# In[ ]:


#print(balanced_df)
import pandas as pd
import matplotlib.pyplot as plt

label_counts = balanced_df['label'].value_counts()

# Create a bar chart
plt.figure(figsize=(6, 4))  # Optional: Set the figure size
label_counts.plot(kind='bar', color=['blue', 'green']) 

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Count of 0s and 1s in "label" column')

plt.show()  # Display the plot


# In[ ]:


import pandas as pd

label_counts = balanced_df['label'].value_counts()

print(label_counts)


# In[ ]:


max_len=100
data = shuffle(balanced_df)  
sentences=balanced_df['text']
labels=balanced_df['label']
len(sentences),len(labels)


# In[ ]:


input_ids=[]
attention_masks=[]

for sent in sentences:
    inp=tokenizer.encode_plus(sent,add_special_tokens = True,max_length =200,pad_to_max_length = True,return_attention_mask = True)
    input_ids.append(inp['input_ids'])
    attention_masks.append(inp['attention_mask'])

input_ids=np.asarray(input_ids)
attention_masks=np.array(attention_masks)
#labels=np.array(labels)
labels=np.array(labels)


# In[ ]:


print('Preparing the pickle file.....')

pickle_inp_path='eert_inp.pkl'
pickle_mask_path='ert_mask.pkl'
pickle_label_path='ert_label.pkl'

pickle.dump((input_ids),open(pickle_inp_path,'wb'))
pickle.dump((attention_masks),open(pickle_mask_path,'wb'))
pickle.dump((labels),open(pickle_label_path,'wb'))


print('Pickle files saved as ',pickle_inp_path,pickle_mask_path,pickle_label_path)


# In[ ]:


print('Loading the saved pickle files..')

input_ids=pickle.load(open(pickle_inp_path, 'rb'))
attention_masks=pickle.load(open(pickle_mask_path, 'rb'))
labels=pickle.load(open(pickle_label_path, 'rb'))
print('Input shape {} Attention mask shape {} Input label shape {}'.format(input_ids.shape,attention_masks.shape,labels.shape))


# In[ ]:


# Set the random state
random_state = 42  # You can use any integer value as the random seed

# Split the data with the specified random state
train_inp, val_inp, train_label, val_label, train_mask, val_mask = train_test_split(
    input_ids, labels, attention_masks, test_size=0.2, random_state=random_state
)

print('Train inp shape {} Val input shape {}\nTrain label shape {} Val label shape {}\nTrain attention mask shape {} Val attention mask shape {}'.format(
    train_inp.shape, val_inp.shape, train_label.shape, val_label.shape, train_mask.shape, val_mask.shape
))


# In[ ]:


# Load the Electra tokenizer and model
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = TFElectraForPreTraining.from_pretrained('google/electra-small-discriminator')


# In[ ]:


# Split data into train and test sets
train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42
)

# Create TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": train_input_ids, "attention_mask": train_attention_masks},
    train_labels
)).shuffle(buffer_size=1000).batch(128)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": test_input_ids, "attention_mask": test_attention_masks},
    test_labels
)).batch(128)


# Running without tunning

# In[ ]:


# Define callbacks
log_dir = 'tensorboard_data/tb_bert'
model_save_path = './models/Electra_model.h5'

# Callback for early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=2,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the best model weights
)

callbacks = [
    ModelCheckpoint(
        filepath=model_save_path, save_weights_only=True,
        monitor='val_loss', mode='min', save_best_only=True
    ),
    TensorBoard(log_dir=log_dir),
    early_stopping  # Add the EarlyStopping callback
]

# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5, epsilon=1e-08)

model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

# Train the model
history = model.fit(
    train_dataset, epochs=40, validation_data=test_dataset, callbacks=callbacks
)


# Running with regularization

# In[ ]:


# Define callbacks
log_dir = 'tensorboard_data/tb_bert'
model_save_path = './models/Electra_model.h5'

# Callback for early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

callbacks = [
    ModelCheckpoint(
        filepath=model_save_path, save_weights_only=True,
        monitor='val_loss', mode='min', save_best_only=True
    ),
    TensorBoard(log_dir=log_dir),
    early_stopping
]

# Define L2 regularization strength
l2_reg_strength = 0.001



# Apply L2 regularization to specific layers (e.g., the classifier layer)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        layer.kernel_regularizer = l2(l2_reg_strength)

# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5, epsilon=1e-08)

model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

# Train the model
history = model.fit(
    train_dataset, epochs=20, validation_data=test_dataset, callbacks=callbacks
)


# Keeping only last layet

# In[ ]:


# Create a new model with only the last layer
last_layer = pretrained_model.get_layer('classifier')
model = tf.keras.Sequential([
    pretrained_model.get_layer('electra'),
    last_layer
])

# Define callbacks
log_dir = 'tensorboard_data/tb_bert'
model_save_path = './models/Electra_model.h5'

# Callback for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    ),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    early_stopping
]

# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5, epsilon=1e-08)

model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

# Train the model
history = model.fit(
    train_dataset,
    epochs=40,
    validation_data=test_dataset,
    callbacks=callbacks
)


# In[ ]:


test_predictions = model.predict(test_dataset)


# In[ ]:


test_predictions_classes = np.argmax(test_predictions.logits, axis=1)

# Get classification report
target_names = ["Hate", "Non hate"]  # Replace with your actual class names
class_report = classification_report(test_labels, test_predictions_classes, target_names=target_names)

print("Classification Report:\n", class_report)

