#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier


# In[16]:


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


# In[17]:


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


# In[18]:


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


# In[19]:


balanced_df.text = balanced_df['text'].apply(data_processing)


# In[20]:


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
df['text']=df['text'].map(preprocess_sentence)


# In[21]:


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


print(balanced_df)


# In[22]:


x = np.array(balanced_df["text"])
y = np.array(balanced_df["label"])


# In[23]:


cv = CountVectorizer()


# In[24]:


X = cv.fit_transform(x)


# In[29]:


from sklearn.metrics import accuracy_score


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


X = balanced_df['text']
y = balanced_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)

# Initialize and train a Naive Bayes classifier (MultinomialNB)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Vectorize the test data using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test_tfidf)

# Calculate accuracy and print it
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print a classification report with precision, recall, and F1-score
print(classification_report(y_test, y_pred))


# In[ ]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
# Create a heatmap with a custom colormap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', 
            xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'],
            cmap='cividis')  # You can change the colormap (e.g., 'coolwarm')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix-MultinomialNB')
plt.show()


# In[ ]:


X = balanced_df['text']
y = balanced_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print a classification report with precision, recall, and F1-score
print(classification_report(y_test, y_pred))

# Get the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:


# Create a heatmap with a custom colormap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', 
            xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'],
            cmap='cividis')  # You can change the colormap (e.g., 'coolwarm')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap-svm_classifier')
plt.show()


# In[ ]:




