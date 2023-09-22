#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraForPreTraining, AdamW
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import unicodedata
import tensorflow_hub as hub
import nltk
from nltk.corpus import stopwords
import keras
from tqdm import tqdm
import pickle
from keras.models import Model, load_model
import keras.backend as K
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import itertools
from sklearn.utils import shuffle
from transformers import BertConfig, BertTokenizer, TFBertModel


# In[ ]:





# In[ ]:


from sklearn.utils import shuffle
df = pd.read_csv(r"C:\Users\Yashika\OneDrive\Desktop\Dissertation\data set\Final Dataset\Racial and ethical hate speech.csv")
df = df.sample(frac=1).reset_index(drop=True) #
df = df[['label','text']]
df = shuffle(df)  
#df['label'] =df['label'].map({ 'nothate':0, 'hate': 1})
df = df[['label','text']]
df.tail()


# In[ ]:


df['label'] =df['label'].map({ 'nothate':0, 'hate': 1})
df = df[['label','text']]
df = shuffle(df)
df.tail()


# In[ ]:


import pandas as pd
from sklearn.utils import resample

minority_class = df['label'].value_counts().idxmin()
print(minority_class)
# Separate majority and minority classes
majority_df = df[df['label'] != minority_class]
minority_df = df[df['label'] == minority_class]

# Downsample the majority class
majority_downsampled = resample(majority_df, replace=False, n_samples=len(minority_df), random_state=100)

# Combine the minority class DataFrame with the downsampled majority class DataFrame
balanced_df = pd.concat([majority_downsampled, minority_df])

# 'balanced_df' now contains a balanced dataset
#print(balanced_df)
import pandas as pd
import matplotlib.pyplot as plt

label_counts = balanced_df['label'].value_counts()

# Create a bar chart
plt.figure(figsize=(6, 4))  # Optional: Set the figure size
label_counts.plot(kind='bar', color=['blue', 'green'])  # Use color to distinguish 0s and 1s

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Count of 0s and 1s in "label" column')

plt.show()  # Display the plot


# In[ ]:


balanced_df = shuffle(balanced_df)
print(balanced_df.tail())


# In[ ]:


X = balanced_df['text']
z =balanced_df['label']
y = z.values


# In[ ]:


balanced_df = balanced_df[['label','text']]
balanced_df = shuffle(balanced_df) 


# In[ ]:


MAX_FEATURES = 200000 # number of words in the vocab


# In[ ]:


vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')


# In[ ]:


vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottlenecks


# In[ ]:


total_length = len(dataset)
train_length = int(total_length * 0.7)
val_length = int(total_length * 0.2)
test_length = int(total_length * 0.1)

# Split the dataset
train = dataset.take(train_length)
val = dataset.skip(train_length).take(val_length)
test = dataset.skip(train_length + val_length).take(test_length)


# In[ ]:


model = Sequential( )
# Create the embedding layer 
model.add(Embedding(MAX_FEATURES+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='relu')))
    # Feature extractor Fully connected layers
#model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer 
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='BinaryCrossentropy', optimizer='Adam')
#history = model.fit(train, epochs=15, validation_data=val)
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy

# Initialize metrics
pre = Precision()
re = Recall()
acc = BinaryAccuracy()  # Use BinaryAccuracy for binary classification

# Compile the model with loss and metrics
model.compile(loss=BinaryCrossentropy(), optimizer='Adam', metrics=[acc])

# Train the model with the specified metrics
history = model.fit(train, epochs=5, validation_data=val)

# After training, you can access the training and validation accuracy from the history
train_accuracy = history.history['binary_accuracy']
val_accuracy = history.history['val_binary_accuracy']


# In[ ]:


true_labels = []
predicted_labels = []

for batch in test.as_numpy_iterator(): 
    # Unpack the batch 
    X_true, y_true = batch
    # Make a prediction 
    yhat = model.predict(X_true)
    
    # Flatten the predictions and convert them to binary labels
    y_true = (y_true.flatten() > 0.5).astype(int)
    yhat = (yhat.flatten() > 0.5).astype(int)
    
    # Append true labels and predicted labels
    true_labels.extend(y_true)
    predicted_labels.extend(yhat)

# Create a classification report
report = classification_report(true_labels, predicted_labels)

print(report)


# In[ ]:


plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Create a heatmap with a custom colormap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', 
            xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'],
            cmap='cividis')  # You can change the colormap (e.g., 'coolwarm')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap- Bidirectional LSTM layer')
plt.show()

