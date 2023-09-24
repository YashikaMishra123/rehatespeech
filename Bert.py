#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import unicodedata
import tensorflow as tf
import tensorflow_hub as hub
import re
import nltk
from nltk.corpus import stopwords
import keras
from tqdm import tqdm
from keras.models import Model
import keras.backend as K
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import itertools
from transformers import TFBertModel, BertConfig,AdamW


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


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'label' is the name of the column in your DataFrame
label_counts = df['label'].value_counts()

# Create a bar chart
plt.figure(figsize=(6, 4))  # Optional: Set the figure size
label_counts.plot(kind='bar', color=['blue', 'green'])  # Use color to distinguish 0s and 1s

# Add labels and title
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Count of 0s and 1s in "label" column')

plt.show()  # Display the plot


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


# In[ ]:


sentences=balanced_df['text']
labels=balanced_df['label']
len(sentences),len(labels)


# In[ ]:


from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig
#num_classes=len(data.label.unique())
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')


# In[ ]:


input_ids=[]
attention_masks=[]

for sent in sentences:
    bert_inp=bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =128,pad_to_max_length = True,return_attention_mask = True)
    input_ids.append(bert_inp['input_ids'])
    attention_masks.append(bert_inp['attention_mask'])

input_ids=np.asarray(input_ids)
attention_masks=np.array(attention_masks)
labels=np.array(labels)


# In[ ]:


print('Preparing the pickle file.....')

pickle_inp_path='./bert_inp.pkl'
pickle_mask_path='./bert_mask.pkl'
pickle_label_path='./bert_label.pkl'

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


# Bert without fine tune

# In[ ]:


history=model.fit([train_inp,train_mask],train_label,batch_size=128,epochs=20,validation_data=([val_inp,val_mask],val_label),callbacks=callbacks)


# MODEL 1 - only for testing and fine tuning

# In[ ]:


# Get the BERT embeddings
bert_output = bert_model([input_ids, train_mask])[0]

# Create a custom head for binary classification
x = Flatten()(bert_output)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=[train_inp, train_mask], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()


# In[ ]:


accumulated_gradients = []
accumulation_steps = 4  # Accumulate gradients over 4 steps

for i, (inputs, labels) in enumerate(train_dataset):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss_value = loss_fn(labels, logits)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    accumulated_gradients.append(gradients)

    if (i + 1) % accumulation_steps == 0:
        # Calculate the average gradient over accumulated steps
        avg_gradients = [tf.reduce_mean(grads, axis=0) for grads in zip(*accumulated_gradients)]
        optimizer.apply_gradients(zip(avg_gradients, model.trainable_variables))
        accumulated_gradients = []  # Clear accumulated gradients


# MODEL 2 - only for testing and fine tuning

# In[ ]:


class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    
    def forward(self, sent_id, mask):
      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      cls_hs = cls_hs.view(-1, cls_hs.size(1))  # Reshape cls_hs
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)


# In[ ]:


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)


# In[ ]:


history=model.fit([train_inp,train_mask],train_label,batch_size=128,epochs=20,validation_data=([val_inp,val_mask],val_label),callbacks=callbacks)


# In[ ]:


model_save_path='./models/bert_model.h5'

trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
trained_model.load_weights(model_save_path)

preds = trained_model.predict([val_inp,val_mask],batch_size=32)
pred_labels = preds.argmax(axis=1)
f1 = f1_score(val_label,pred_labels)
print('F1 score',f1)
print('Classification Report')
print(classification_report(val_label,pred_labels,target_names=target_names))

print('Training and saving built model.....')   


# In[ ]:


test_predictions_classes = np.argmax(preds.logits, axis=1)

# Get classification report
target_names = ["Hate", "Non hate"]  
class_report = classification_report(val_label, test_predictions_classes, target_names=target_names)

print("Classification Report:\n", class_report)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


conf_matrix = confusion_matrix(val_label, test_predictions_classes)

# Create a heatmap with a custom colormap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', 
            xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'],
            cmap='cividis') 
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()


# In[ ]:


plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()

