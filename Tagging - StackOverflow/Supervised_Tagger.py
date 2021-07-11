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


# Create list with all "real" tags and create list of all tags for each individual

Data['Tag_Bags'] = 0

Tags = []

for i in range(len(Data)):
  Personal_Tags = []
  for h in range(len(Data['Tags'].iloc[i])):
    if Data['Tags'].iloc[i][h] == '<':
      Beginning = h+1
      for k in range(h, len(Data['Tags'].iloc[i])):
        if Data['Tags'].iloc[i][k] == '>':
          End = k
          Personal_Tags.append(Data['Tags'].iloc[i][Beginning:End])
          Tags.append(Data['Tags'].iloc[i][Beginning:End])
          break
  Data['Tag_Bags'].iloc[i] = Personal_Tags
  


# Reorder all Tags by frequency

UniqueTags = list(set(Tags))
Number = []

for i in range(len(UniqueTags)):
  Number.append(Tags.count(UniqueTags[i]))

zipped_lists = zip(Number, UniqueTags)
sorted_pairs = sorted(zipped_lists, reverse=True)
tuples = zip(*sorted_pairs)   
Number, UniqueTags = [list(tuple) for tuple in  tuples]

# Store all tags that appear at least 100 times

TopTags = UniqueTags[0:187]


Data['Clean_Tags'] = 0

for i in range(len(Data)):
  Individual_Bag = []
  for j in range(len(Data['Tag_Bags'].iloc[i])):
    if (Data['Tag_Bags'].iloc[i][j] in TopTags):
      Individual_Bag.append(Data['Tag_Bags'].iloc[i][j])
  Data['Clean_Tags'].iloc[i] = Individual_Bag

  
# Count vectorizer

from sklearn.feature_extraction.text import CountVectorizer

Corpus = []

for i in range(len(Data)):
  Corpus.append(str(Data['Clean_Tags'].iloc[i]))

count_vectorizer = CountVectorizer(analyzer='word',stop_words= 'english')
X = count_vectorizer.fit_transform(Corpus)

  # Store Count vectorizer in a DataFrame

Targets_Data = pd.DataFrame(data = X.toarray(),columns = count_vectorizer.get_feature_names())

  # Replace all Counts by a Binary

for i in range(len(Targets_Data.columns)):
  VariableName = str(Targets_Data.columns[i])
  Targets_Data[VariableName][Targets_Data[VariableName] > 0] = 1




# Fusion des 500 features avec les 200 variables targets

Count_Data = Count_Data.iloc[:,0:456]

Count_Data_Renamed = Count_Data

for i in range(len(Count_Data.columns)):
    Count_Data_Renamed[str(i)] = Count_Data.iloc[:,i]
    
Count_Data_Renamed = Count_Data_Renamed.iloc[:,456:912]

Supervised_Data = pd.concat([Count_Data_Renamed, Targets_Data], axis=1)

Count_Data = Count_Data.iloc[:,0:456]


# Separation des données en Test et Trainnning

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

Shuffled_Data = shuffle(Supervised_Data)

X = Shuffled_Data.iloc[:,0:456]
y = Shuffled_Data.iloc[:,456::]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

Supervised_train = pd.concat([X_train, y_train], axis=1)
Supervised_test = pd.concat([X_test, y_test], axis=1)



# Fit 200 logistic regressions - one for each Tag - on the 500 features, and store in dictionary

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

X = Supervised_train.iloc[:,0:456]

LogitModels = {}
LogitReg = {}
for i in range(456,len(Supervised_Data.columns)):
    LogitReg["Logit_{0}".format(i)] = LogisticRegression(random_state=0)
    y = Supervised_train.iloc[:,i]
    LogitModels["model_{0}".format(i)] = LogitReg["Logit_{0}".format(i)].fit(X, y)



from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X = Supervised_train.iloc[:,0:456]

ForestModels = {}
ForestReg = {}
for i in range(456,len(Supervised_train.columns)):
    ForestReg["Forest_{0}".format(i)] = RandomForestClassifier(n_estimators=100)
    y = Supervised_train.iloc[:,i]
    ForestModels["model_{0}".format(i)] = ForestReg["Forest_{0}".format(i)].fit(X, y)
      


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
  
# Define Tagging function

def ForestLogit_Tagger(Title, Text):
  Test, Test_Bag = Map_to_Vectorizer(Title, Text)
  Test_Data = pd.DataFrame(data = Test,columns = list(Count_Data.columns))
  
  Tags = []

  for i in range(456,len(Supervised_Data.columns)):
    LogitVariableName = "Logit_" + str(Supervised_Data.columns[i])
    ForestVariableName = "Forest_" + str(Supervised_Data.columns[i])
    Test_Data[LogitVariableName] = LogitReg["Logit_{0}".format(i)].predict(Test_Data.iloc[:,:456])
    Test_Data[ForestVariableName] = ForestReg["Forest_{0}".format(i)].predict(Test_Data.iloc[:,:456])
    if Test_Data[LogitVariableName].iloc[0] > 0.4:
        Tags.append(Supervised_Data.columns[i])
    if Test_Data[ForestVariableName].iloc[0] > 0.4:
        Tags.append(Supervised_Data.columns[i])
    Tags = list(set(Tags))
  return Tags




import gradio as gr
  
gr.Interface(fn=ForestLogit_Tagger, title="Generating Tags - Supervised", description="Cette API génère des tags en 'mappant' le texte dans l'espace des 500 features obtenu par le biais d'un Count Vectorizer, puis en en prédisant tour à tour si chacun des 187 tags les plus populaires doivent-être associé au texte. La fonction renvoie les tags associés à une prédiction positive.", inputs=["text","textbox"], outputs="textbox").launch()

