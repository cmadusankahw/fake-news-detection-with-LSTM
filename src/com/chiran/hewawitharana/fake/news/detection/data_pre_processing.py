import pandas as pd
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import gensim

# TASK #1: DATA LOADING AND PRE PROCESSING

# load the data
df_true = pd.read_csv("data/True.csv")
df_fake = pd.read_csv("data/Fake.csv")

df_true.head()
df_fake.head()

# check for null values
df_true.isnull().sum()
df_fake.isnull().sum()

df_true.info()
df_fake.info()

# add a target class column to indicate whether the news is real or fake
df_true['isfake'] = 1
df_true.head()

df_fake['isfake'] = 0
df_fake.head()

# Concatenate Real and Fake News
df = pd.concat([df_true, df_fake]).reset_index(drop=True)
df.head()

# drop unnecessary Date column # RUN ONLY ONCE
df.drop(columns=['date'], inplace=True)

# combine title and text together
df['original'] = df['title'] + ' ' + df['text']
df.head()

print(df['original'][0])

# # TASK #2: PERFORM DATA CLEANING

# download stopwords
nltk.download("stopwords")

# Obtain additional stopwords from nltk
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# Remove stopwords and remove words with 2 or less characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)

    return result


# Apply the function to the dataframe
df['clean'] = df['original'].apply(preprocess)

# Show original news
print(df['original'][0])


# Show cleaned up news after removing stopwords
print(df['clean'][0])
df.head()


# Obtain the total words present in the dataset
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)

print("List of Words :", list_of_words)
print("List of Words: Length :", len(list_of_words))

# Obtain the total number of unique words
total_words = len(list(set(list_of_words)))
print("Total Words",total_words)

# join the words into a string
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

df.head()
print(df['clean_joined'][0])
