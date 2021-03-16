# Import libraries
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
EN = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import CountVectorizer

# Import dataset
df = pd.read_csv('question.csv')

df.isnull().sum()
df.isna().sum()
df.nunique()

# To select which column to process
df['is_equal']= (df['title']==df['question'])

# Check if Slug has duplicates - if same questions are repeated
dupdf = df[df.duplicated(['slug'],keep=False)]

#Clean data for exploration
'''questions without tags'''
(df['tags']=='[]').value_counts()
type(df['tags'])
df0 = df[df['tags'] != '[]']

'''Removing brackets from tags'''
df['tags'] = df['tags'].apply(lambda x: x.replace('[','').replace(']','').replace("'",'')) 

'''Make a dict having tag frequencies'''
df.tags = df.tags.apply(lambda x: x.split(','))
tag_freq_dict = {}
for tags in df.tags:
    for tag in tags:
        if tag not in tag_freq_dict:
            tag_freq_dict[tag] = 0
        else:
            tag_freq_dict[tag] += 1
      
# Visualize all tags based on their frequencies using a WordCloud
wordcloud = WordCloud(background_color='black',
                      width=1600,
                      height=800,
                     ).generate_from_frequencies(tag_freq_dict)
fig = plt.figure(figsize=(30,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

'''Removing unnecessary characters from Title'''
def preprocess(title):
    title = title.str.replace("(<br/>)", "") #line break
    title = title.str.replace('(<a).*(>).*(</a>)', '') 
    title = title.str.replace('(&amp)', '') #symbols
    title = title.str.replace('(&gt)', '')
    title = title.str.replace('(&lt)', '')
    title = title.str.replace('(\xa0)', ' ')  #non-breaking space
    return title
df['qnData'] = preprocess(df['title'])
df['word_count'] = df['qnData'].apply(lambda x: len(str(x).split()))

# Visualize the distribution of question word count
df['word_count'].plot(
    kind='hist',
    bins=100,
    rwidth=1.0,
    title='Question Word Count Distribution')

# Visualize Questions with highest views
print(df.groupby('title').sum()['views'].sort_values(ascending=False).head(15))

# Visualize the most frequently visited Tags
df0.groupby('tags').sum()['views'].sort_values(ascending=False).head(15).plot(
    kind='bar',x='tags', title='Most frequently visited Tags')

# The distribution of top unigrams before removing stop words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['qnData'], 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['qnData' , 'count'])
df1.groupby('qnData').sum()['count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words in Question Corpus before removing stop words')

# The distribution of top unigrams after removing stop words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['qnData'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['qnData' , 'count'])
df2.groupby('qnData').sum()['count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words in Question corpus after removing stop words')

# The distribution of top bigrams before removing stop words
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['qnData'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['qnData' , 'count'])
df3.groupby('qnData').sum()['count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 bigrams in Question corpus before removing stop words')

# The distribution of top bigrams after removing stop words
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['qnData'], 20)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['qnData' , 'count'])
df4.groupby('qnData').sum()['count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 bigrams in Question corpus after removing stop words')

# Count of views per Category
df.groupby('category').sum()['views'].sort_values(ascending=False).plot(
    kind='pie',x='category', title='Count of views per category')