#importing the required packages
import pandas as pd
df=pd.read_csv('C:/Users/nomesh.palakaluri.EMEA/OneDrive - Drilling Info/Desktop/Text model/Emotion.csv')
text=df['Text'].values.astype(str)
emotion=df['Emotion'].values.astype(str)
print(text)

#data cleaning 
import re
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

#processing the data cleaning on the dataset
m=[]
for i in range(len(text)):
    s=text[i]
    preprocess_text(s)
    m.append(s)
print(m)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(m)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, emotion, test_size=0.2, random_state=42)

#loading the testing dataset
#testing the output on a dataset
df1=pd.read_csv('C:/Users/nomesh.palakaluri.EMEA/OneDrive - Drilling Info/Desktop/Text model/emotion_detect/Testing data/tweet_emotions.csv')
df1['content']=df1['content'].apply(preprocess_text)
df1['content']

#testing over the entire dataset
#testing the data
for i in range(len(df1['content'])):
    sen1=df1['content'][i]
    sen1=vectorizer.transform(sen1.split())
    classifier.predict(sen1)
    b = Counter(classifier.predict(sen1))
    print (b.most_common()[0][0])
    print(i)
