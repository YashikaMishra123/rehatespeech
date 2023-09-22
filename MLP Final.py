#!/usr/bin/env python
# coding: utf-8

# In[153]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import ElectraTokenizer, ElectraForPreTraining, AdamW, BertTokenizer, TFBertModel, BertConfi
import tensorflow_hub as hub
from tqdm import tqdm
import pickle
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import itertools
from sklearn.utils import shuffle
# import the required libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import unicodedata


# In[154]:


df = pd.read_csv(r"C:\Users\Yashika\OneDrive\Desktop\Dissertation\data set\Final Dataset\Racial and ethical hate speech.csv")
df = df.sample(frac=1).reset_index(drop=True) #
df = df[['label','text']]
df = shuffle(df)  
#df['label'] =df['label'].map({ 'nothate':0, 'hate': 1})
df = df[['label','text']]
df.tail()


# In[155]:


from sklearn.utils import shuffle
# Load your dataset
from sklearn.utils import shuffle
df = pd.read_csv(r"C:\Users\Yashika\OneDrive\Desktop\Dissertation\data set\Final Dataset\Racial and ethical hate speech.csv")
df = df.sample(frac=1).reset_index(drop=True) #
df = df[['label','text']]
df = shuffle(df)  
#df['label'] =df['label'].map({ 'nothate':0, 'hate': 1})
df = df[['label','text']]
df.tail()
      
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
df['label'] =df['label'].map({ 'nothate':0, 'hate': 1})
df = df[['label','text']]
df = shuffle(df)
df.head()


# In[156]:


from nltk.tokenize import word_tokenize
#creating a function to process the data
def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','',tweet)
    tweet = re.sub(r'ð','',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)
balanced_df.text = balanced_df['text'].apply(data_processing)


# In[157]:


import unicodedata
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    pro = unicode_to_ascii(w.lower().strip())
    pro = re.sub(r"([?.!,¿])", r" ", w)
    pro = re.sub(r'[" "]+', " ", w)
    pro = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    pro=clean_stopwords_shortwords(w)
    pro=re.sub(r'@\w+', '',w)
    return pro
balanced_df['text']=df['text'].map(preprocess_sentence)
balanced_df = shuffle(balanced_df)  
balanced_df.tail()


# In[159]:


balanced_df.tail()
balanced_df['label'] =balanced_df['label'].map({ 'nothate':0, 'hate': 1})


# In[160]:


texts = balanced_df['text'].tolist()
labels = balanced_df['label'].tolist()


# In[161]:


# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)


# In[162]:


sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)


# In[163]:


# Encode the string labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


# In[164]:


# One-hot encode the labels
one_hot_labels = keras.utils.to_categorical(labels)


# In[165]:


# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences, 
                                                one_hot_labels, 
                                                test_size=0.3)


# In[166]:


# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, 
                    output_dim=128, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=128, activation="sigmoid"))
model.add(Dense(units=2, activation="sigmoid"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(xtrain, ytrain, epochs=1, batch_size=16, validation_data=(xtest, ytest))


# In[167]:


from sklearn.metrics import classification_report

y_true = np.argmax(ytest, axis=1)

# Convert one-hot encoded y_pred back to class labels
y_pred = np.argmax(model.predict(xtest), axis=1)

# Generate the classification report
report = classification_report(y_true, y_pred)

print(report)


# In[172]:


# Create the confusion matrix
confusion = confusion_matrix(y_true, y_pred)

# Display the confusion matrix
print(confusion)


# In[175]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_true, y_pred)

# Create a heatmap with a custom colormap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', 
            xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'],
            cmap='cividis')  # You can change the colormap (e.g., 'coolwarm')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap- MLP')
plt.show()


# In[ ]:




