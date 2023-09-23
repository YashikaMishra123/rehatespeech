#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout, Input
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt
import itertools
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
from transformers import *
from transformers import TFBertModel,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig


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
#df = shuffle(df)
df.tail()


# In[ ]:


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


# In[ ]:


balanced_df = shuffle(balanced_df)
print(balanced_df.tail())


# In[ ]:


dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


# In[ ]:


max_len=300
sentences=balanced_df['text']
labels=balanced_df['label']
len(sentences),len(labels)


# Model 1

# In[ ]:


def create_model():
    inps = Input(shape = (max_len,), dtype='int64')
    masks= Input(shape = (max_len,), dtype='int64')
    dbert_layer = dbert_model(inps, attention_mask=masks)[0][:,0,:]
    dense = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.02))(dbert_layer)
    dropout= Dropout(0.5)(dense)
    pred = Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.02))(dropout)
    model = tf.keras.Model(inputs=[inps,masks], outputs=pred)
    print(model.summary())
    return model   


# In[ ]:


def create_model():
    inps = Input(shape = (max_len,), dtype='int64')
    masks= Input(shape = (max_len,), dtype='int64')
    dbert_layer = dbert_model(inps, attention_mask=masks)[0][:,0,:]
    dense = Dense(512,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(dbert_layer)
    dropout= Dropout(0.5)(dense)
    pred = Dense(1, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)
    model = tf.keras.Model(inputs=[inps,masks], outputs=pred)
    print(model.summary())
    return model   


# Model 2

# In[ ]:


def create_model():
    inps = Input(shape=(max_len,), dtype='int64')
    masks = Input(shape=(max_len,), dtype='int64')
    dbert_layer = dbert_model(inps, attention_mask=masks)[0][:, 0, :]
    dense = Dense(512, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(dbert_layer)
    dropout = Dropout(0.5)(dense)
    pred = Dense(1, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout)
    
    # Add additional layers here
    x = Dense(128, activation='relu')(pred)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    model = tf.keras.Model(inputs=[inps, masks], outputs=x)  # Output from the additional layers
    print(model.summary())
    return model


# model 3

# In[ ]:


def create_model():
    inps = Input(shape = (max_len,), dtype='int64')
    masks= Input(shape = (max_len,), dtype='int64')
    dbert_layer = dbert_model(inps, attention_mask=masks)[0]
     # Apply a global average pooling layer to summarize token embeddings
    avg_pooling = tf.keras.layers.GlobalAveragePooling1D()(dbert_layer)
     # Apply a global average pooling layer to summarize token embeddings
     # Add a dropout layer to prevent overfitting
    dropout = Dropout(0.5)(avg_pooling)
    # Add a dense layer with ReLU activation
    dense = Dense(512,activation='sigmoid',kernel_regularizer=regularizers.l2(0.02))(dropout)
    # Add another dropout layer
    dropout2 = Dropout(0.2)(dense)
    
    #dropout= Dropout(0.5)(dense)
    pred = Dense(2, activation='softmax',kernel_regularizer=regularizers.l2(0.02))(dropout2)
    model = tf.keras.Model(inputs=[inps,masks], outputs=pred)
    #
  
    return model
print(model.summary())
  


# In[ ]:


input_ids=[]
attention_masks=[]

for sent in sentences:
    dbert_inps=dbert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =max_len,pad_to_max_length = True,return_attention_mask = True,truncation=True)
    input_ids.append(dbert_inps['input_ids'])
    attention_masks.append(dbert_inps['attention_mask'])

input_ids=np.asarray(input_ids)
attention_masks=np.array(attention_masks)
labels=np.array(labels)


# In[ ]:


print('Preparing the pickle file.....')

pickle_inp_path='dbert_inp.pkl'
pickle_mask_path='dbert_mask.pkl'
pickle_label_path='.dbert_label.pkl'

pickle.dump((input_ids),open(pickle_inp_path,'wb'))
pickle.dump((attention_masks),open(pickle_mask_path,'wb'))
pickle.dump((labels),open(pickle_label_path,'wb'))
print('Pickle files saved as ',pickle_inp_path,pickle_mask_path,pickle_label_path)


# In[ ]:


print('Loading the saved pickle files..')

input_ids=pickle.load(open(pickle_inp_path, 'rb'))
attention_masks=pickle.load(open(pickle_mask_path, 'rb'))
labels=pickle.load(open(pickle_label_path, 'rb'))


# In[ ]:


train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,labels,attention_masks,test_size=0.2)

print('Train inp shape {} Val input shape {}\nTrain label shape {} Val label shape {}\nTrain attention mask shape {} Val attention mask shape {}'.format(train_inp.shape,val_inp.shape,train_label.shape,val_label.shape,train_mask.shape,val_mask.shape))


log_dir='dbert_model'
model_save_path='./dbert_model.h5'

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_dir)]



# In[ ]:


loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metric = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

model.compile(loss=loss, optimizer=optimizer, metrics=[metric])


# In[ ]:


callbacks= [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_dir)]
model.compile(loss=loss,optimizer=optimizer, metrics=[metric])


# In[ ]:


history=model.fit([train_inp,train_mask],train_label,batch_size=128,epochs=30,validation_data=([val_inp,val_mask],val_label),callbacks=callbacks)


# In[ ]:


trained_model = create_model()
trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
trained_model.load_weights(model_save_path)


# In[ ]:


preds = trained_model.predict([val_inp,val_mask],batch_size=128)
pred_labels = preds.argmax(axis=1)
f1 = f1_score(val_label,pred_labels)
f1


# In[ ]:


print('F1 score',f1)
print('Classification Report')
print(classification_report(val_label,pred_labels,target_names=target_names))

print('Training and saving built model.....')   


# In[ ]:


print(history.history)
from matplotlib import pyplot as plt
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()


# In[ ]:


conf_matrix = confusion_matrix(val_label,pred_labels)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:


import seaborn as sns
# Create a heatmap with a custom colormap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', 
            xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'],
            cmap='cividis')  # You can change the colormap (e.g., 'coolwarm')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap-Decision tree classifer')
plt.show()

