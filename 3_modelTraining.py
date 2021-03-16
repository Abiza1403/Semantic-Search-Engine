import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import multiprocessing
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from time import time
import matplotlib.pyplot as plt
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')

df=pd.read_csv('question.csv')

# Join contents of Title and tags 
df['Data'] = df['title'].str.cat(df['tags'], sep =' ')

# Converting all the words to lower case
df['text_lower']=df['Data'].str.lower()

# Removing all the special characters 
df['spl_char_rem']=df['text_lower'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', ' ', x))).astype(str)
rem_punc = string.punctuation   
def remove_punctuation(x):
    return x.translate(str.maketrans('', '', rem_punc))
df['spl_char_rem']=df['spl_char_rem'].apply(lambda x: remove_punctuation(x)).astype(str)

# Removing numbers 
df['numbers_rem']=df['spl_char_rem'].apply((lambda x: re.sub(r'\d', '', x))).astype(str)

# Lemmatization
w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])
df['lemmatized'] = df.numbers_rem.apply(lemmatize_text)

# Removing words with < 3 characters
df['ch_rem']=df['lemmatized'].apply((lambda x: re.sub(r'\W*\b\w{1,2}\b', '', x))).astype(str)

# Removing extra white spaces
df['whitespace_rem']=df['ch_rem'].apply((lambda x: re.sub(r'\s+', ' ', x))).astype(str)

# Removing Stopwords and Tokenization
stop = stopwords.words('english')
def identify_tokens(row):
    question = row['whitespace_rem']
    tokens = nltk.word_tokenize(question)
    token_words = [w for w in tokens if not w in stop]
    return token_words
df['tokened'] = df.apply(identify_tokens, axis=1)
print("List of lists. Let's confirm: ", type(df['tokened']), " of ", type(df['tokened'][0]))

# Create sentence with tokens
tbwd = TreebankWordDetokenizer()
df['qn']=df['tokened'].apply(lambda x: tbwd.detokenize(x)).astype(str)

# Create URL column
df['title_url'] = 'https://ask.learncbse.in/t/' + df['slug'] + '/' + df['id'].astype(str)

# Dropping unnecessary columns
cols_to_drop = ['question', 'slug', 'category', 'tags', 'Data', 'text_lower', 'spl_char_rem', 'numbers_rem', 'whitespace_rem', 'lemmatized', 'ch_rem']
df.drop(cols_to_drop, inplace=True, axis = 1)

# Word2Vec - CBOW model
cores = multiprocessing.cpu_count()
tokens = df['tokened']
model = Word2Vec(size=300, window=10, min_count=2, workers=cores-1)
t = time()
model.build_vocab(tokens, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
t = time()
model.train(tokens, total_examples=model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
model.init_sims(replace=True)

# Testing the model
model.wv.most_similar(positive=['human'], topn=5)

model.wv.similarity(w1='social',w2='revolution')

model.wv.doesnt_match(['algebra', 'tower', 'matrix'])

# Exploring the trained model
word_vectors = model.wv
len(word_vectors.vocab)
vector = model.wv['algorithm']  # vector of a word
len(vector)

# 2D representation of word embeddings using t-SNE
wanted_words = []
count = 0
for word in word_vectors.vocab:
    if count<300:
        wanted_words.append(word)
        count += 1
    else:
        break
wanted_vocab = dict((k, word_vectors.vocab[k]) for k in wanted_words if k in word_vectors.vocab)

t = time()
X = model[wanted_vocab] 
tsne_model = TSNE(perplexity=40, n_components=2, init="pca", n_iter=5000, random_state=23)
Y = tsne_model.fit_transform(X)
print('Time to train model on t-SNE: {} mins'.format(round((time() - t) / 60, 2)))

#Plot the t-SNE output
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(Y[:, 0], Y[:, 1])
words = list(wanted_vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))
ax.set_yticklabels([]) #Hide ticks
ax.set_xticklabels([]) #Hide ticks
plt.show()


#df.to_csv('/Users/abiza/Documents/Projects/Semantic search Engine/cbse/QList.csv', index = False, header=True)
#model.save('/Users/abiza/Documents/Projects/Semantic search Engine/cbse/w2vModel.bin')

















