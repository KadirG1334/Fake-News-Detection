import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_len=(len(train))
test_len = (len(test))


#concatenate both dataframes
df = pd.concat([train, test], axis=0)
df.reset_index(drop=True,inplace = True)

train = df.iloc[:train_len,:]
test = df.iloc[train_len:,:]
test = test.drop(columns=['label'])

x_test = test.drop(columns=['id','keyword','location'])
################### NLP ######################
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#nltk.download('stopwords')
ps = WordNetLemmatizer()
stopwords = stopwords.words('english')
#nltk.download('wordnet')
def cleaning_data(row):
    
    # convert text to into lower case
    row = row.lower() 
    
    # this line of code only take words from text and remove number and special character using RegX
    row = re.sub('[^a-zA-Z]' , ' ' , row)
    
    # split the data and make token.
    token = row.split() 
    
    # lemmatize the word and remove stop words like a, an , the , is ,are ...
    news = [ps.lemmatize(word) for word in token if not word in stopwords]  
    
    # finaly join all the token with space
    cleannedNews = ' '.join(news) 
    
    # return cleanned news
    return cleannedNews 
################### NLP ######################


x = train.drop(columns=['id','keyword','location','label'])
y = train['label']

df['text'] = df['text'].apply(lambda x : cleaning_data(x))
x['text'] = x['text'].apply(lambda x : cleaning_data(x))
x_test['text'] = x_test['text'].apply(lambda x : cleaning_data(x))




################ VECTORIZER ################
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 8000 , lowercase=False , ngram_range=(1,2))

vecTrainData = vectorizer.fit_transform(x['text'])
vecTrainData = vecTrainData.toarray()
#print(vecTrainData)
vecPredData = vectorizer.fit_transform(x_test['text'])
print(vecPredData)
#herhalde fit diyerek uygun hale getiriyruz
vecPredData = vecPredData.toarray()
#text formatından array formatına getirmemizi sağlıyo vectorizer
#böylece convert hatası almıyoruz 
#array formatına da gelince labek ve texti model.fit deyip fitleyebiliyoruz
newTrainingData = pd.DataFrame(vecTrainData , columns=vectorizer.get_feature_names())#_out
newPredictData =  pd.DataFrame(vecPredData , columns=vectorizer.get_feature_names())#_out
#print(newTrainingData)
#print(newPredictData)
################  VECTORIZER ################
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(newTrainingData,y)
pred = model.predict(vecPredData)
#print(pred)
#print(len(pred))

###### WRITE TO CSV ##########
submission = pd.DataFrame()
submission['id']=test['id']
submission['text']=test['text']#x_test['test']
submission['label'] = pred
submission.to_csv('newSubmisson.csv',index=False)



"""özet olarak 7 bin civarı verili bir data vardı elimizde  burdakı txt formatında bilgilerle
labellar eşleştirilmişti ki bu labellarda text te bulunan haberin doğru veya yanlış olduğu bilgisi
vardı,nihayetinde elimize 3 bin civarı bir versi seti ile önceden train ettiğimzi modelin
bu 3 bin verilik datayı da predict etmesini amaçladık sonrasında da bu tahmini csv ye yazdırdık"""