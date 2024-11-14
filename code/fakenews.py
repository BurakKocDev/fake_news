import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import accuracy_score , ConfusionMatrixDisplay , classification_report , roc_curve

# Printing the stopwords in English
print(stopwords.words('english'))



df = pd.read_csv('train.csv')
print(f"The shape of the dataset is: {df.shape}")
print(df.head(5))
print(df.info())
print(df.isnull().sum())
df.fillna(" ", inplace= True)
df['content'] = df['title'] + " " + df['author']
print(df.head())

port_stem = PorterStemmer()
def stemming(content):
    
    stemmed_content= re.sub('[^a-zA-Z]',' ',content)
    
    stemmed_content = stemmed_content.lower() 
    
    stemmed_content = stemmed_content.split()
    
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content
df['content']= df['content'].apply(stemming)
df['content']



transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(df['content'].values)
tfidf = transformer.fit_transform(counts)

targets = df['label'].values
print(f"target shape: {targets.shape}")
print(f"X shape: {tfidf.shape}")

X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, test_size=0.2, random_state=49)
print(f"The shape of X_train is: {X_train.shape[0]}")
print(f"The shape of X_test is: {X_test.shape[0]}")


#itarate

def train(model , model_name):
    model.fit(X_train,y_train)
    print(f"Training accuracy of {model_name} is {model.score(X_train,y_train)}")
    print(f"testing accuracy of {model_name} is {model.score(X_test,y_test)}")
def conf_matrix(model):
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )
def class_report(model):
    print(classification_report(
        y_test,
        model.predict(X_test)
    ))
    
    
#LogisticRegression
model_lr = LogisticRegression()
train(model_lr, 'LogisticRegression')


conf_matrix(model_lr)

class_report(model_lr)


#SVM
svc_model= SVC()
train(svc_model, 'SVM')
class_report(svc_model)


#DecisionTreeClassifier
depth_num= range(50, 71, 2)
training_acc= []
testing_acc = []
for depth in depth_num:
    tree_model = DecisionTreeClassifier(max_depth=depth,random_state=42)
    tree_model.fit(X_train,y_train)
    training_acc.append(tree_model.score(X_train,y_train))
    testing_acc.append(tree_model.score(X_test,y_test))
print("Training Accuracy Scores:", training_acc[:3])
print("testing Accuracy Scores:", testing_acc[:3])


plt.plot(depth_num , training_acc , label= 'Training')
plt.plot(depth_num , testing_acc , label= 'Testing')
plt.xlabel('Max_Depth')
plt.ylabel('Accuracy_score')
plt.legend();


# final model
tree_final_model=DecisionTreeClassifier(max_depth=58,random_state=42)
tree_final_model.fit(X_train,y_train)
tree_training_acc = tree_final_model.score(X_train,y_train)
tree_testing_acc = tree_final_model.score(X_test,y_test)
print(f"Training accuracy of DesicionTreeClassifier is {tree_training_acc}")
print(f"testing accuracy of DesicionTreeClassifier is {tree_testing_acc}")

conf_matrix(tree_final_model)
class_report(tree_final_model)    


#RandomForestClassifier
clf= RandomForestClassifier(random_state=42)
params={
    "n_estimators": range(50,125,25),
    "max_depth": range(60,81,2)
}
params
{'n_estimators': range(50, 125, 25), 'max_depth': range(60, 81, 2)}
rfc_model = GridSearchCV(
    clf,
    param_grid= params,
    cv= 5,
    n_jobs= -1,
    verbose=1
)
rfc_model.fit(X_train,y_train)


cv_results= pd.DataFrame(rfc_model.cv_results_)
print(cv_results.sort_values('rank_test_score').head(10))


rfc_model.best_params_
rfc_model.predict(X_test)

acc_train = rfc_model.score(X_train , y_train)
acc_test = rfc_model.score(X_test , y_test)

print(f"Training accuracy: {round(acc_train , 4)}")
print(f"test accuracy: {round(acc_test , 4)}")


conf_matrix(rfc_model)
class_report(rfc_model)


models = pd.DataFrame({
    
    "Models": ["Logestic Regression" , "SVM", "DecisionTreeClassifier","RandomForestClassifier"],
    "Score":[model_lr.score(X_test,y_test) ,svc_model.score(X_test,y_test) ,tree_testing_acc,acc_test ]
    
})
models.sort_values(by="Score" , ascending=False)



colors= ['gray' , 'pink','red','green']
sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
sns.barplot(x=models['Models'],y=models['Score'], palette=colors )
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Model Selection")
plt.show();


print("best model decision tree classifier")


import joblib

"""# En iyi modeli kaydetme (örneğin, DecisionTreeClassifier)
best_model = DecisionTreeClassifier(max_depth=58, random_state=42)
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'best_model.pkl')"""


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Metin sütununu (text) alıyoruz
corpus = df['text'].astype(str)

# CountVectorizer 
count_vectorizer = CountVectorizer(ngram_range=(1, 2))  # 1-2 gram aralığı
X_count = count_vectorizer.fit_transform(corpus)

# TF-IDF dönüştürücüsü
tfidf_transformer = TfidfTransformer(smooth_idf=False)
X_tfidf = tfidf_transformer.fit_transform(X_count)


import joblib
joblib.dump(count_vectorizer, 'count_vectorizer.pkl')
joblib.dump(tfidf_transformer, 'tfidf_transformer.pkl')

print("CountVectorizer ve TfidfTransformer kaydedildi.")
