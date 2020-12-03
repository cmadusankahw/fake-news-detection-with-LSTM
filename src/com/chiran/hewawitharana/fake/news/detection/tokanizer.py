import nltk
nltk.download('punkt')
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# split data into test and train
from sklearn.model_selection import train_test_split

# importing previously pre processed df
from data_pre_processing import df, total_words

# TASK #6: PREPARE THE DATA BY PERFORMING TOKENIZATION AND PADDING

x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size=0.2)

# Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

print("The encoding for document\n", df.clean_joined[0], "\n is : ", train_sequences[0])

# Add padding can either be maxlen = 4406 or smaller number maxlen = 40 seems to work well based on results
padded_train = pad_sequences(train_sequences, maxlen=40, padding='post', truncating='post')
padded_test = pad_sequences(test_sequences, maxlen=40, truncating='post')

for i, doc in enumerate(padded_train[:2]):
    print("The padded encoding for document", i + 1, " is : ", doc)