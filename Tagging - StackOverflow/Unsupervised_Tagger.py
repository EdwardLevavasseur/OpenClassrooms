# Define Text Cleaning function which eliminates punctuation and figures

def TextCleaning(Text):
  Text = str(Text)
  Text = Text.replace('_', ' ')
  Text = Text.replace('-', ' ')
  Text = Text.replace('+', ' ')
  Text = Text.replace("'", ' ')
  Text = Text.replace('"', '')
  Text = Text.replace('”', '')
  Text = Text.replace('(', '')
  Text = Text.replace(')', '')
  Text = Text.replace('[', '')
  Text = Text.replace(']', '')
  Text = Text.replace('{', '')
  Text = Text.replace('}', '')
  Text = Text.replace('>', ' ')
  Text = Text.replace('<', ' ')
  Text = Text.replace('@', '')
  Text = Text.replace('|', '')
  Text = Text.replace('/', '')
  Text = Text.replace('"', '')
  Text = Text.replace('=', '')
  Text = Text.replace('.', '')
  Text = Text.replace(',', '')
  Text = Text.replace(':', '')
  Text = Text.replace('!', '')
  Text = Text.replace('?', '')
  Text = Text.replace('0', '')
  Text = Text.replace('1', '')
  Text = Text.replace('2', '')
  Text = Text.replace('3', '')
  Text = Text.replace('4', '')
  Text = Text.replace('5', '')
  Text = Text.replace('6', '')
  Text = Text.replace('7', '')
  Text = Text.replace('8', '')
  Text = Text.replace('9', '')
  Text = Text.replace(' p ', '')
  return Text


# Define Bagging function which transforms text into a clean bag of words

import nltk
nltk.download('punkt')
nltk.download('wordnet')

def BaggingText(Text):

  # Tokenize text

  Bag = nltk.word_tokenize(Text)

  # Make the words cleaner

  lemma = nltk.wordnet.WordNetLemmatizer()

  for i in range(0,len(Bag)):
    Bag[i] = Bag[i].lower()
    Bag[i] = lemma.lemmatize(Bag[i])

  return Bag


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Import Data

import pandas as pd

Data = pd.read_csv('/home/edward/QueryResults_Clean2.csv', sep=',', engine='python', error_bad_lines=False)



# Count vectorizer

from sklearn.feature_extraction.text import CountVectorizer

Corpus = []

for i in range(len(Data)):
  Corpus.append(str(Data['Clean_Bags'].iloc[i]))

count_vectorizer = CountVectorizer(analyzer='word',stop_words= 'english')

X = count_vectorizer.fit_transform(Corpus)

  # Store Count vectorizer in a DataFrame

Count_Data = pd.DataFrame(data = X.toarray(),columns = count_vectorizer.get_feature_names())


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

X = Count_Data

lda = LatentDirichletAllocation(n_components=20,
                                random_state=0)
lda.fit(X)

# get topics for some given samples:
Count_LDA =  lda.transform(X)

Count_LDA = pd.DataFrame(data = Count_LDA)



# Define function that maps 

def Map_to_Vectorizer(Text, Title):
  CleanText = TextCleaning(Text)
  CleanText = BaggingText(CleanText)
  CleanTitle = TextCleaning(Title)
  CleanTitle = BaggingText(CleanTitle)
  Bag = []
  for i in range(len(CleanText)):
    if CleanText[i] not in stopwords.words('english'):
      Bag.append(CleanText[i])
  for j in range(len(CleanTitle)):
    if CleanTitle[j] not in stopwords.words('english'):
      Bag.append(CleanTitle[j])
  Count = []
  for h in range(len(list(Count_Data.columns)[0:456])):
    Count.append(Bag.count(list(Count_Data.columns)[h]))
  Number = []
  Number.append(Count)
  return Number, Bag




def Unsupervised_Tagger(Title, Text):
  Test, Test_Bag = Map_to_Vectorizer(Title, Text)
  Test_Data = pd.DataFrame(data = Test,columns = Count_Data.columns)
  X = Test_Data
  Test_LDA =  lda.transform(X)
  Test_LDA = pd.DataFrame(data = Test_LDA)
  zipped_lists = zip(list(Test_LDA.iloc[0]), list(Test_LDA.columns))
  sorted_pairs = sorted(zipped_lists, reverse=True)
  tuples = zip(*sorted_pairs)

  LDA, Categories = [list(tuple) for tuple in  tuples]


  Tags = []

  for i,topic in enumerate(lda.components_):
    if (i == Categories[0]) & (LDA[0] > 0.9):
      bla = [count_vectorizer.get_feature_names()[i] for i in topic.argsort()[-3:]]
      for j in range(len(bla)):
          Tags.append(bla[-j-1])
    for h in range(0,3):
      if (i == Categories[h]) & (LDA[h] > 0.3):
        bla = [count_vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]]
        for j in range(len(bla)):
          if bla[-j-1] in Test_Bag:
            Tags.append(bla[-j-1])

  Tags = list(set(Tags))

  return Tags


import gradio as gr

gr.Interface(fn=Unsupervised_Tagger, title="Generating Tags - Unsupervised", description="Cette API génère des tags en 'mappant' le texte dans l'espace des 20 topics générés par le modèle de Latent Dirichlet Allocation (LDA), paramétré sur 50.000 questions de Stack Overflow.", inputs=["text","textbox"], outputs="textbox").launch()



