from flask import Flask,render_template,request
app = Flask(__name__)
import pandas as pd
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer 
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# file1=open('model.pkl','rb')
# clf = pickle.load(file1)
# file1.close()

@app.route('/',methods=["GET","POST"])
def prob():
    if request.method=="POST":
        # try:
            # fe=request.form
            # fever2=float(fe['fever'])
            # fever=round(fever2)
            # age=int(fe['age'])
            # pain=int(fe['pain'])
            # nose=int(fe['nose'])
            # breath=int(fe['breath'])

            # user_input=[[fever,age,pain,nose,breath]]
            fe=request.form
            query1=fe['question']

            data=pd.read_csv("combine1.csv",encoding = "ISO-8859-1")
            data2=pd.read_csv("links.csv",encoding = "ISO-8859-1")

            df1=data[['QueryText','KccAns','QueryType']]
            df2=data2[['QueryType','links']]

            
            nOfTrainQs = len(df1['QueryText'])

            trainFullQ1s = []

            for i, q in enumerate(df1['QueryText']):
                print("Cleaning Train Q1s: {0:.2f}".format(
                float(i) / nOfTrainQs), end='\r')
                trainFullQ1s.append(text_to_wordlist(q,True,True))

            nOfTrainQs2 = len(df1['QueryType'])
            qd = []
            for i, q in enumerate(df1['QueryType']):
                
                qd.append(q)

            qd2 = []
            for i, q in enumerate(df2['QueryType']):
                qd2.append(q)

            ld2 = []
            for i, q in enumerate(df2['links']):
                ld2.append(q)

            nOfTrainQs2 = len(df1['QueryType'])
            qd = []
            for i, q in enumerate(df1['QueryType']):
                qd.append(q)

            corpus = trainFullQ1s 

            bow_vectorizer, bow_features = bow_extractor(corpus)
            features = bow_features.todense()
            feature_names = bow_vectorizer.get_feature_names()
            # def display_features(features, feature_names):
            #     df = pd.DataFrame(data=features, columns=feature_names)


            transformer = TfidfTransformer(smooth_idf=False)
            count = features
            tfidf = transformer.fit_transform(count)
            X2=tfidf.toarray()

            queryTFIDF = TfidfVectorizer().fit(corpus)
            
            # query1=input("enter a query:")
            query1=text_to_wordlist2(query1,True,True)
            queryTFIDF = queryTFIDF.transform([query1])


            cosine_similarities = cosine_similarity(queryTFIDF, X2).flatten()
            related_question_indices = cosine_similarities.argsort()[:-20:-1]
            a=related_question_indices[:20]

            flag=True
            answers=[]
            for i in range(1,11):
                
                ans=str(df1.KccAns[a[i]])
                
                if ans=='nan':
                    cat=df1.QueryType[a[i]].lower() 
                    for j in qd2:
                        if j.lower() in cat:
                            ind=qd2.index(j)
                            if flag:
                                print(f"{i}  you may get detailed information in {ld2[ind]}\n")
                                answers.append([f"{i}  you may get detailed information in the link\n",ld2[ind]])    
                                flag= False
                                break
                else:
                    print(str(i)  +"." +ans+"\n")
                    answers.append([str(i)  +"." +ans+"\n",''])


          
            return render_template('result.html',answers=answers)
        # except:
        #     return render_template('index.html')
            
    return render_template('index.html')



def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = str(text).lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # Return a list of words
    return(text)


def bow_extractor(corpus, ngram_range=(1,1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def display_features(features, feature_names):
    df = pd.DataFrame(data=features, columns=feature_names)

def text_to_wordlist2(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # Return a list of words
    return(text)










if __name__ == "__main__":
    app.run(debug=True)
