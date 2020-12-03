import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
nltk.download('punkt')
import plotly.express as px

# importing previously pre processed df
from data_pre_processing import df

# TASK #2: DATA VISUALIZATION AND UNDERSTANDING

df.head()

# plot the number of samples in 'subject'
plt.figure(figsize=(8, 8))
sns.countplot(y="subject", data=df)

# plot the number of samples in 'isfake'
plt.figure(figsize = (8, 8))
sns.countplot(y="isfake", data=df)

# plot the word cloud for text that is Real
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=2000, width=1600, height=800, stopwords=stop_words).generate(
    " ".join(df[df.isfake == 1].clean_joined))
plt.imshow(wc, interpolation='bilinear')

# plot the word cloud for text that is Fake
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=2000, width=1600, height=800, stopwords=stop_words).generate(
    " ".join(df[df.isfake == 0].clean_joined))
plt.imshow(wc, interpolation='bilinear')

# return all the tokens (each and every single word) as an array of tokens
nltk.word_tokenize(df['clean_joined'][1])

# length of maximum document will be needed to create word embeddings
maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if maxlen < len(tokens):
        maxlen = len(tokens)
print("The maximum number of words in any document is =", maxlen)

# visualize the distribution of number of words in a text
# interactive visualizations with ploty
fig = px.histogram(x = [len(nltk.word_tokenize(x)) for x in df.clean_joined], nbins = 100)
fig.show()
