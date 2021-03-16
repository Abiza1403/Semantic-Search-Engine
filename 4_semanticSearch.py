# import libraries
import numpy as np
import pandas as pd
import gensim
import spacy
import re
from nltk.corpus import stopwords
EN = spacy.load('en_core_web_sm')
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser

# Load file
df=pd.read_csv('/Users/abiza/Documents/Projects/Semantic search Engine/cbse/QList.csv')

# Load pre-trained embeddings
model = gensim.models.word2vec.Word2Vec.load('/Users/abiza/Documents/Projects/Semantic search Engine/cbse/w2vModel.bin')

# model sanity check
model.wv.most_similar(positive=['human'], topn=5)
model.wv.similarity(w1='social',w2='revolution')
model.wv.doesnt_match(['algebra', 'tower', 'matrix'])

# Treating missing values
df['qn'] = df['qn'].replace(np.nan, '', regex=True)

# Calculate Sentence Embeddings
def question_to_vec(question, embeddings, dim=300):
    question_embedding = np.zeros(dim)
    valid_words = 0
    for word in question.split(' '):
        if word in embeddings:
            valid_words += 1
            question_embedding += embeddings[word]
    if valid_words > 0:
        return question_embedding/valid_words
    else:
        return question_embedding
    
qn_sent_embeddings = []
for qn in df.qn:
    qn_sent_embeddings.append(question_to_vec(qn, model))
qn_sent_embeddings = np.array(qn_sent_embeddings)
embeddings = pd.DataFrame(data = qn_sent_embeddings)

# General operation for search string
def tokenize_text(text):
    "Apply tokenization using spacy to docstrings."
    tokens = EN.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------  
    
# Search Output
search_string = 'tangent'
search_string = ' '.join(normalize(tokenize_text(search_string)))
results_returned = '30'
search_vect = np.array([question_to_vec(search_string, model)])    # Vectorize the user query

# Calculate Cosine similarites for the query and all questions
cosine_similarities = pd.Series(cosine_similarity(search_vect, qn_sent_embeddings)[0])

# Write the results to a HTML page
output =""
for i,j in cosine_similarities.nlargest(int(results_returned)).iteritems():
    output += '<a target="_blank" href='+ str(df.title_url[i])+'><h2>' + df.title[i] + '</h2></a>'
    output += '<h3> Similarity Score: ' + str(j) + '</h3>'
    output += '<h3> No of views: ' + str(df.views[i]) + '</h3>'
    output +='<p style="font-family:verdana; font-size:110%;"> '
    for i in df.qn[i][:50].split():
        if i.lower() in search_string:
            output += " <b>"+str(i)+"</b>"
        else:
            output += " "+str(i)
    output += "</p><hr>"
    
output = '<h3>Results:</h3>'+output

f = open('Search Results.html','w')
f.write(output)
f.close()

#Open HTML from Python
filename = 'file:///Users/abiza/Documents/Projects/Semantic search Engine/cbse/' + 'Search Results.html'
webbrowser.open_new_tab(filename)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 

# Tangent
# iron sinks?
# energy-source?
# online coaching

