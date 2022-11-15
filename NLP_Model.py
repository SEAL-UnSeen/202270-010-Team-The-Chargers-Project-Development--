#%%-----------------------------Install----------------------------------------
#pip install spacy_sentence_bert
#pip install git+https://github.com/MartinoMensio/spacy-sentence-bert.git
#pip install spacy
#pip uninstall wasabi -y
#pip install -U sentence-transformers --user
#pip install numpy
#pip install pandas
#pip install wasabi==0.9.1
#pip install https://github.com/MartinoMensio/spacy-sentence-bert/releases/download/v0.1.2/en_stsb_distilbert_base-0.1.2.tar.gz#en_stsb_distilbert_base-0.1.2 --user
#pip install https://github.com/MartinoMensio/spacy-sentence-bert/releases/download/v0.1.2/en_stsb_bert_large-0.1.2.tar.gz#en_stsb_bert_large-0.1.2 --user
#pip install https://github.com/MartinoMensio/spacy-sentence-bert/releases/download/v0.1.2/en_stsb_roberta_large-0.1.2.tar.gz#en_stsb_roberta_large-0.1.2 --user
#pip install numpy --user
#pip install textblob
#pip install autocorrect
#pip install wordninja
#pip install textstat 
#pip install inflect
#pip install lemminflect
#pip install dtreeviz
#pip install pandas-visual-analysis
!pip install -U scikit-learn
#pip install xgboost
#%%-----------------------------Import-----------------------------------------

import os
import numpy as np
import seaborn as sns
import pandas as pd
import spacy_sentence_bert
import matplotlib.pyplot as plt
import textblob
from autocorrect import Speller 
from textblob import TextBlob
import wordninja
import textstat 
import re
import inflect
import nltk
import spacy

#%%-----------------------------Get_wd_&_file----------------------------------

cwd = os.getcwd()
print(cwd)

df = pd.read_excel('SOPs v1.xlsx')

#df = pd.read_excel('SOPsv1Clean.xlsx')
#print(df)
#print(df.label.unique())
#print(df.Final_vector)
#df = df.dropna()
#df.Final_vector.dtypes

#%%-----------------------------Data_Cleansing---------------------------------

#-----------------------------------------Replacing Abbreviations/Accronyms----

replacers = {"l/g":"landing gear", "hsc-manual":"high speed counter manual", "vnav":"vertical navigation", "lnav":"lateral navigation", "econ":"optimum descent speed", "flx":"reduced takeoff thrust", "mct":"maximum continuous thrust", "mcp":"maximum continuous power", "n1":"cockpit gauge which presents the rotational speed of the low pressure", "to/ga":"take-off go around", "v/s":"stalling speed", "g/s":"ground Stop", "spd ":"speed mode", "flch":"flight level change", "alt":"altitude", "pth":"path", "atc":"Air traffic control", "ovrd ctr":"overdrive control traffic zone", "fl":"flight level", "navaids":"navigational Aids", "mcdu":"multi-function control and display unit", "fma":"flight mode annunciator", "hyd":"hydraulic", "rmps":"risk management process", "hdg":"heading the direction", "loc":"loss of aircraft control", "thr ref":"thrust reference", "cmd":"Command", "v1":"maximum speed at which a rejected takeoff can be done", "cdu":"control display units", "egt ":"exhaust gases temperature", "conf ":"configuration", "apu":"auxiliary power unit", "aft":"towards the rear", "pnf":"pilot not flying", "pf":"pilot flying", "c":"captain", "pfd":"primary flight display", "f/o":"first officer", "egt":"temperature of the exhaust gases", "pu":"processing unit", "cf/o":"captain flying", "nd":"navigation display", "dh/mda":"referenced to mean sea level or aerodrome elevation ", "gpws":"Ground Proximity Warning System", "a/skid":"skid", "hf":"high frequency", "vhf":"very high frequency", "fac 1":"flight augmentation computer", "f-pln":"flight plan", "fcu":"fuel control unit", "mcduperf clb":"take off Mode", "nw strg disc":"nose wheel steering locked", "ldg elev":"landing elevation", "emer elec gen ":"emergency electric generator", "fuel x feed":"fuel cross feed", "f-pln":"flight plan", "ext pwr":"external power", "gen":"generator", "sysoff":"system off", "sd":"serial dail", "atvr":"Automated Transfer Vehicle", "to":"take-off", "go ":"go arround", "ead":"Electronic Attitude and Direction ", "eadcheck":"Electronic Attitude and Direction check", "n2":"rotational speed of the high pressure engine spool", "fcp":"Final Circulating Pressure", "cdu/fmc":"Control Display Unit flight management computer", "fms":"Flight Management System", "vors":"Very High Frequency Omni-Directional Range", "ilss":"Instrument Landing System", "ils":"Instrument Landing System", "ndb":"non-directional beacon", "ndbs":"non-directional beacon", "canc/rcl":"Cancel/Recall", "eicascaution":"Engine Indicating and Crew Alerting System", "eicas":"Engine Indicating and Crew Alerting System", "l/r":"left or right", "aux":"auxiliary", "trans":"transmitter", "tk":"tank", "tnk":"tank", "ram":"using the airflow created by a moving object to increase ambient pressure", "spd sel":"speed select", "selvfe":"maximum flap extended speed", "atcnotify":"air traffic control notify", "extractovrd":"extract override", "fl100":"flight level 100", "fac":"Flight Augmentation Computer", "flx/mct":"max continuous thrust or reduced takeoff thrust", "flx":"reduced takeoff thrust", "mct":"max continuous thrust", "v2":"Takeoff Safety Speed", "wing + ctr":"wings and center", "ap/fd":"Airborne Collision Avoidance System", "ap/fdoff":"Airborne Collision Avoidance System off", "a/thr":"automatic throttle", "a/throff":"automatic throttle off", "l/gup":"landing gear up", "l/gdown":"landing gear down", "fl":"flight level", "toga":"take-off go around", "grnd splrs":"ground spoilers", "agl":"above ground level", "ecam":"Electronic Centralized Aircraft Monitor", "clb/clb":"climb/climb", "thr ":"Throttle", "clb/op":"Open Climb", "clb":"climb", "fmgs":" Flight Management Guidance System", "ta/ra":"Traffic Advisory/Resolution Advisory", "TARA":"Traffic Advisory/Resolution Advisory", "mda/dh ":"Minimum Descent Altitude/Height ", "mdh/dh ":"Minimum Descent Altitude/Height ", "mdh":"Minimum Descent Altitude/Height ", "mda":"Minimum Descent Altitude/Height ", "v/s":"velocity speeds", "ias":"Indicated Airspeed ", "irs":"inertial reference system", "ins":"Inertial Navigation System", "spd":"speed"}

df['text2'] = (df.text.str.replace('[...…]','')
    .str.split()
    .apply(lambda x: ' '.join([replacers.get(e, e) for e in x])))
        
#---------------------------Filter labels--------------------------------------

df = df[df['label'] != "Decide" ]

#---------------------------Filter labels based on low count-------------------

df = df.groupby('label').filter(lambda x : (x['label'].count()>=50).any())

#---------------------------Remove Stop Words----------------------------------
#BERT algorithm doesnt suggest to remove Stop Words, accuracy decreases as well when we remove them.

#from nltk.corpus import stopwords
#stop = stopwords.words('english')

#df['text2'] = df['text2'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        
#--------------------------------Sentiment-------------------------------------

df['sentiment'] = df['text2'].apply(lambda x: TextBlob(x).sentiment)

#----------------------------Spell corrector-----------------------------------

df['text2'] = df['text2'].apply(lambda x:str(TextBlob(x).correct()))


#------------------Remove short strings in text--------------------------------

df = df[df['text2'].str.len()>3]

#--------------------Replace Special Characters--------------------------------

df['text2'] = df['text2'].replace(r'[^\w\s]|_', '', regex=True)

#-----------------------Remove single Character--------------------------------

df['text2'] = df['text2'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')

#-----------------------Remove extra spaces------------------------------------

df['text2'] = df['text2'].replace(r'\s+', ' ', regex=True)

#-----------------------Remove single space------------------------------------

df['text2'] = df['text2'].apply(lambda x: str.lstrip(x))

#-----------------------Word separation----------------------------------------

df['text2'] = df['text2'].apply(lambda x: wordninja.split(x))

df['text2'] = df['text2'].apply(lambda x: " ".join(x))

#-----------------------text stats---------------------------------------------

df['Readability_Index'] = df['text2'].apply(lambda x: textstat.automated_readability_index(x))
df['Reading_Time'] = df['text2'].apply(lambda x: textstat.reading_time(x))

#-----------------------Extract first word of text-----------------------------

df['word'] = df['text2'].str.split(' ').str[0]

#-----------------------Parts of Speech Tagging--------------------------------

df['full_tag'] = df['text2'].apply(lambda x: TextBlob(x).tags)

#-------------------Parts of Speech Tagging2-----------------------------------


from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV, getAllLemmas

df['tag2'] = df['word'].apply(lambda x: getAllLemmas(x))

#------Categorical Label to Numerical Label Predifined labels------------------

#df["label_code1"]  = {'label': [ 'Action (How) ',  'Action (What) ',  'Action (Where) ',  'Actor ',  'Decide (How) ',  'Decide (What) ',  'Decide (Where) ',  'Trigger (How) ',  'Trigger (What) ',  'Trigger (Where) ',  'Verification (How) ',  'Verification (What) ',  'Verification (Where) ',  'Waiting (How) ',  'Waiting (What) ',  'Waiting (Where) '], 'label_code': [ '9 ',  '7 ',  '8 ',  '0 ',  '6 ',  '4 ',  '5 ',  '3 ',  '1 ',  '2 ',  '15 ',  '13 ',  '14 ',  '12 ',  '10 ',  '11 ']}	

#One hot enconding


#----------------------Label Automatic labels----------------------------------

from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
df["label_code1"] = ord_enc.fit_transform(df[["label"]])


#------------------filter readability index------------------------------------

df = df[df['Readability_Index'] > -9] 

#-----------Conditional lable to indentify model-------------------------------

#------------------------------Vector 2----------------------------------------
df['Condition_1'] = np.where((df['word']=='if') | (df['word']=='after') | (df['word']=='before') |(df['word']=='when') | (df['word']=='prior') | (df['word']=='during') | (df['word']=='until') |(df['word']=='while') | (df['word']=='following') | (df['word']=='every') | (df['word']=='verify') |(df['word']=='observe') | (df['word']=='check') | (df['word']==''),"v2", "v1")
df['Condition_1_labels'] = np.where((df['label']=='Decide (What)') | (df['label']=='Trigger (What)') | (df['label']=='Verification (What)') | (df['label']=='Waiting (What)'),"Conditional label","No condition")
df['Eliminate'] = np.where((df['Condition_1'] =='v2') & (df['Condition_1_labels']=='No condition'),"Yes", "No")
df = df[df['Eliminate'] != "Yes" ]
df['dic_len']  = df['tag2'].apply(len)
df['Feature3'] =  df['tag2'].apply(lambda x: str(x))



#----------------------------------Vector 3------------------------------------
Verb_1 = 'VERB'
Noun_1= 'NOUN'


df['Condition_2'] = df['Feature3'].str.findall(Verb_1 or (Noun_1 and Verb_1) , flags = re.IGNORECASE)
df['Condition_2'] = df['Condition_2'].astype('string')
df['Condition_2']= df['Condition_2'].astype(str).str.replace(r'\[|\]|', '')
df['Condition_2']= df['Condition_2'].astype(str).str.replace(r'\'|', '')
df['Condition_3'] = np.where((df['Condition_2'] == "VERB") & (df['label']=='Action (How)'),'Verb',0)
                             
df.to_excel (r'C:\Users\EstebanEchandi\Desktop\SOPsv1Clean1.xlsx', index = False, header=True)

#Unbalance
#SMOTE
#ADASYN                      
                             


#%%-----------------------------BERT_Word_Vectors------------------------------

#nlp = spacy_sentence_bert.load_model('en_stsb_bert_large')
#nlp = spacy_sentence_bert.load_model('en_stsb_distilbert_base')

nlp = spacy_sentence_bert.load_model('en_stsb_roberta_large')

df['vector'] = df['text2'].apply(lambda x: nlp(x).vector)
df['vector2'] = df['word'].apply(lambda x: nlp(x).vector)
df['vector3'] = df['Condition_3'].apply(lambda x: nlp(x).vector)


#df['vector'] = nlp(df['text2'].values).vector
#df['vector2'] = nlp(df['word'].values).vector
#df['vector3'] = nlp(df['Condition_3'].values).vector

#%%-----------------------------Final_Vector_Model-----------------------------

df['Final_vector']  = np.where((df['Condition_1'] == "v2"), df["vector2"], df["vector"])

df['Final_vector']  = np.where((df['Condition_3'] == 'Verb'), df['vector3'], df['Final_vector'])

col = ['label', 'Final_vector']
df1 = df[col]


#%%-----------------------------Save_clean_file--------------------------------

df.to_excel (r'C:\Users\EstebanEchandi\Desktop\SOPsv1Clean.xlsx', index = False, header=True)

#%%-----------------------------Frecuencies_Lengths_Statistics-----------------

counts = df['label2'].value_counts()
counts.plot(kind='bar', legend=False, grid=True, figsize=(8, 5))

counts = df['label'].value_counts()
counts.plot(kind='bar', legend=False, grid=True, figsize=(8, 5))

lens = df.text.str.len()
lens.hist(bins = np.arange(0,200,5))

df['word'].describe()
df['label'].describe()
df['label_code'].describe()
df['Procedure'].describe()


#%%-----------------------------Dataset_split----------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Final_vector'].tolist(), df['label'].tolist(), test_size=0.3, random_state=690)

#-----------------------------Additional_labels--------------------------------

from io import StringIO

df['category_id'] = df['label'].factorize()[0]
category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)

features = df['Final_vector'].values
features = features.tolist()
features = np.array(features)

labels = df.category_id


#%%-----------------------------Model_SVC--------------------------------------

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(C=5, gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

#--------------------------classification_report-------------------------------

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)

#--------------------------Confusion_matrix------------------------------------
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

classes = range(15)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGn)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#%%-----------------------------Model_RandomForest-----------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=0)

params = {'n_estimators': [1, 10, 50, 100, 1000],
    'criterion': ["gini", "entropy", "log_loss"],
    'max_depth': [1, 10, 50, 100, 1000]}

clf = GridSearchCV(
    estimator=model, 
    param_grid=params, 
    cv=10,  
    n_jobs=-1)

clf.fit(X_train, y_train)

cv_results = pd.DataFrame(clf.cv_results_)
cv_results.head()
cv_results = cv_results[['mean_test_score', 'param_criterion',  'param_max_depth']]
cv_results.sort_values(by='mean_test_score', ascending=False)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#--------------------------classification_report-------------------------------

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)


#%%-----------------------------Model_XGBoost----------------------------------

from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

le = LabelEncoder()
y_train2 = le.fit_transform(y_train)
y_test2 = le.fit_transform(y_test)


clf = XGBClassifier(booster = 'gblinear',
 learning_rate =0.1,
 n_estimators=100,
 max_depth=1,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 seed=27)

params = {'n_estimators': [1, 10, 100],
      'booster': ['gbtree', 'gblinear','dart'],
          'max_depth': [1],
    'objective': ["multi:softmax"]}

clf = GridSearchCV(
    estimator=model, 
    param_grid=params, 
    cv=10,  
    n_jobs=-1)

clf.fit(X_train, y_train2)

cv_results = pd.DataFrame(clf.cv_results_)
cv_results.head()
cv_results = cv_results[['mean_test_score', 'param_booster', 'param_objective', 'param_n_estimators','param_max_depth']]
cv_results.sort_values(by='mean_test_score', ascending=False)

predictions = clf.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test2,predictions))

clf_report = classification_report(y_test2,predictions,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)

#%%-----------------------------Model_Decision trees---------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

model = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

params = {'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 10, 100, 1000]}

clf = GridSearchCV(
    estimator=model, 
    param_grid=params,
    cv=10,  
    n_jobs=-1)

clf.fit(X_train, y_train)

cv_results = pd.DataFrame(clf.cv_results_)
cv_results.head()
cv_results = cv_results[['mean_test_score', 'param_criterion', 'param_splitter', 'param_max_depth']]
cv_results.sort_values(by='mean_test_score', ascending=False)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)

#%%-----------------------------Model_Summary----------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
  
  
models = [
    RandomForestClassifier(n_estimators=500, max_depth=20, random_state=0),
    #LinearSVC(),
   SVC(C=5, gamma='auto'),
    #MultinomialNB(),
    DecisionTreeClassifier(),
    LogisticRegression(random_state=0),
    XGBClassifier(booster = 'gblinear', learning_rate =0.1,  n_estimators=100, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8, objective= 'multi:softmax',  nthread=4, seed=27)  ]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])  
  
import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df, width = .5)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=6, jitter=True, edgecolor="gray", linewidth=1)

plt.show()  


#%%-----------------------------Test_model-------------------------------------

Examples = ['the takeoff must becontinued', 'Action how verb' ,'flight crew']

label = ['Action (What)', 'Action (How)' ,'Actor']

for Examples, label in zip(Examples, label):
  print(Examples,)
  print(f"True Label: {label}, Predicted Label: {clf.predict(nlp(Examples).vector.reshape(1, -1))[0]} \n")










#%%-----------------------------Working_Part_1---------------------------------

# One hot enconding
from sklearn import preprocessing
y_train2 = preprocessing.label_binarize(y_train2, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)
gsearch1.fit(X_train, y_train2)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_








import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

trains = df


target = "label_code1"
IDcol = "Final_vector"

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['label_code1'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['label_code1'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label_code1'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

FV = trains['Final_vector'].tolist()
FV =train['Final_vector'].to_numpy(dtype='object', copy=bool, na_value=np.nan)  
FV = np.asarray(FV)



























FV.dtypes

#Choose all predictors except target & IDcols

print(FV2)predictors = FV
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, dtrain['label_code1'], predictors)


from keras.datasets import imdb
from keras.preprocessing import sequence
from sequence_classifiers import CNNSequenceClassifier


df1 = df[["label_code1", 'Final_vector']]

!pip install seglearn
import seglearn as sgl

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


data = sgl.load_watch()

X_train, X_test, y_train, y_test = train_test_split(df['Final_vector'].tolist(), df['label'].tolist(), test_size=0.3, random_state=690)



from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(df['Final_vector'].tolist(), df['label'].tolist(), test_size=0.3, random_state=690)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
pipe.score(X_test, y_test)




from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


X_train, X_test, y_train, y_test = train_test_split(df['Final_vector'], df['label'], test_size=0.3, random_state=690)
count_vect = CountVectorizer()

X_train = [str (item) for item in X_train]
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)




df1['Final_vector2'] = [str (item) for item in df1['Final_vector']]

df1['Final_vector2'] =df1['Final_vector'].loc[0,0]

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, df1['Final_vector'], df1['label_code1'], scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


#%%-----------------------------Working_Part_2---------------------------------

df1['category_id'] = df1['label'].factorize()[0]
from io import StringIO
category_id_df = df1[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)


df1.head()



import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df1.groupby('label').label.count().plot.bar(ylim=0)
plt.show()


#from sklearn.feature_extraction.text import TfidfVectorizer

#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

#features = tfidf.fit_transform(df.text2).toarray()
features = df1['Final_vector'].values
features = features.tolist()
features = np.array(features)
labels = df1.category_id
features.dtype



from sklearn.feature_selection import chi2
import numpy as np

N = 2
for label, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(label))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
  

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(features , df1['label'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train, y_train)

print(clf.predict(count_vect.transform(["if switch is on"])))
  
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
  
  
models = [
    RandomForestClassifier(n_estimators=500, max_depth=20, random_state=0),
    #LinearSVC(),
   # SVC(gamma='auto'),
    #MultinomialNB(),
    #DecisionTreeClassifier(),
    #LogisticRegression(random_state=0),
    XGBClassifier(booster = 'gblinear', learning_rate =0.1, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8, objective= 'multi:softmax',  nthread=4, seed=27)  ]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])  
  
import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()  

#%%-----------------------------Working_Part_3---------------------------------

from sklearn.svm import SVC
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(df['Final_vector'].values):
    print(train_index, test_index)
    X_train, X_test = df['Final_vector'].iloc[train_index], df['Final_vector'].iloc[test_index]
    y_train, y_test = df['label'].iloc[train_index], df['label'].iloc[test_index]
    
    X_train = X_train.tolist()
    y_train = y_train.tolist()
    X_test = X_test.tolist()
    y_test = y_test.tolist()
    
    clf = SVC(C=5, gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))

    clf_report = classification_report(y_test,y_pred,output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True) 


#c values check

#grid search cv
X_train.shape
y_train.shape