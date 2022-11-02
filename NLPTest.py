#----------------------------------------Install-------------------------------
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

#----------------------------------------Import--------------------------------
# Import 
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

#-----------------------------------------Get wd and excel --------------------

cwd = os.getcwd()
print(cwd)

# read by default 1st sheet of an excel file
df = pd.read_excel('SOPs v1.xlsx')


print(df)
print(df.label.unique())
print(df.isnull().sum())
df = df.dropna()
df.dtypes


#-----------------------------------------Data Cleansing-----------------------

#-----------------------------------------Replacing Abbreviations/Accronyms----

replacers = {"l/g":"landing gear", "hsc-manual":"high speed counter manual", "vnav":"vertical navigation", "lnav":"lateral navigation", "econ":"optimum descent speed", "flx":"reduced takeoff thrust", "mct":"maximum continuous thrust", "mcp":"maximum continuous power", "n1":"cockpit gauge which presents the rotational speed of the low pressure", "to/ga":"take-off go around", "v/s":"stalling speed", "g/s":"ground Stop", "spd ":"speed mode", "flch":"flight level change", "alt":"altitude", "pth":"path", "atc":"Air traffic control", "ovrd ctr":"overdrive control traffic zone", "fl":"flight level", "navaids":"navigational Aids", "mcdu":"multi-function control and display unit", "fma":"flight mode annunciator", "hyd":"hydraulic", "rmps":"risk management process", "hdg":"heading the direction", "loc":"loss of aircraft control", "thr ref":"thrust reference", "cmd":"Command", "v1":"maximum speed at which a rejected takeoff can be done", "cdu":"control display units", "egt ":"exhaust gases temperature", "conf ":"configuration", "apu":"auxiliary power unit", "aft":"towards the rear", "pnf":"pilot not flying", "pf":"pilot flying", "c":"captain", "pfd":"primary flight display", "f/o":"first officer", "egt":"temperature of the exhaust gases", "pu":"processing unit", "cf/o":"captain flying", "nd":"navigation display", "dh/mda":"referenced to mean sea level or aerodrome elevation ", "gpws":"Ground Proximity Warning System", "a/skid":"skid", "hf":"high frequency", "vhf":"very high frequency", "fac 1":"flight augmentation computer", "f-pln":"flight plan", "fcu":"fuel control unit", "mcduperf clb":"take off Mode", "nw strg disc":"nose wheel steering locked", "ldg elev":"landing elevation", "emer elec gen ":"emergency electric generator", "fuel x feed":"fuel cross feed", "f-pln":"flight plan", "ext pwr":"external power", "gen":"generator", "sysoff":"system off", "sd":"serial dail", "atvr":"Automated Transfer Vehicle", "to":"take-off", "go ":"go arround", "ead":"Electronic Attitude and Direction ", "eadcheck":"Electronic Attitude and Direction check", "n2":"rotational speed of the high pressure engine spool", "fcp":"Final Circulating Pressure", "cdu/fmc":"Control Display Unit flight management computer", "fms":"Flight Management System", "vors":"Very High Frequency Omni-Directional Range", "ilss":"Instrument Landing System", "ils":"Instrument Landing System", "ndb":"non-directional beacon", "ndbs":"non-directional beacon", "canc/rcl":"Cancel/Recall", "eicascaution":"Engine Indicating and Crew Alerting System", "eicas":"Engine Indicating and Crew Alerting System", "l/r":"left or right", "aux":"auxiliary", "trans":"transmitter", "tk":"tank", "tnk":"tank", "ram":"using the airflow created by a moving object to increase ambient pressure", "spd sel":"speed select", "selvfe":"maximum flap extended speed", "atcnotify":"air traffic control notify", "extractovrd":"extract override", "fl100":"flight level 100", "fac":"Flight Augmentation Computer", "flx/mct":"max continuous thrust or reduced takeoff thrust", "flx":"reduced takeoff thrust", "mct":"max continuous thrust", "v2":"Takeoff Safety Speed", "wing + ctr":"wings and center", "ap/fd":"Airborne Collision Avoidance System", "ap/fdoff":"Airborne Collision Avoidance System off", "a/thr":"automatic throttle", "a/throff":"automatic throttle off", "l/gup":"landing gear up", "l/gdown":"landing gear down", "fl":"flight level", "toga":"take-off go around", "grnd splrs":"ground spoilers", "agl":"above ground level", "ecam":"Electronic Centralized Aircraft Monitor", "clb/clb":"climb/climb", "thr ":"Throttle", "clb/op":"Open Climb", "clb":"climb", "fmgs":" Flight Management Guidance System", "ta/ra":"Traffic Advisory/Resolution Advisory", "TARA":"Traffic Advisory/Resolution Advisory", "mda/dh ":"Minimum Descent Altitude/Height ", "mdh/dh ":"Minimum Descent Altitude/Height ", "mdh":"Minimum Descent Altitude/Height ", "mda":"Minimum Descent Altitude/Height ", "v/s":"velocity speeds", "ias":"Indicated Airspeed ", "irs":"inertial reference system", "ins":"Inertial Navigation System", "spd":"speed"}


df['text2'] = (df.text.str.replace('[...…]','')
    .str.split()
    .apply(lambda x: ' '.join([replacers.get(e, e) for e in x])))
        
#*************************Filter labels*************************

df = df[df['label'] != "Decide" ]
        
#*************************Sentiment *************************
df['sentiment'] = df['text2'].apply(lambda x: TextBlob(x).sentiment)

#*************************Spell corrector*************************

df['text2'] = df['text2'].apply(lambda x:str(TextBlob(x).correct()))


#*************************Remove short strings in text*************************

df = df[df['text2'].str.len()>3]

#*************************Replace Special Characters*************************

df['text2'] = df['text2'].replace(r'[^\w\s]|_', '', regex=True)

#*************************Remove single Character*************************

df['text2'] = df['text2'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')

#*************************Remove extra spaces*************************

df['text2'] = df['text2'].replace(r'\s+', ' ', regex=True)

#*************************Remove single space*************************

df['text2'] = df['text2'].apply(lambda x: str.lstrip(x))

#*************************Word separation*************************

df['text2'] = df['text2'].apply(lambda x: wordninja.split(x))

df['text2'] = df['text2'].apply(lambda x: " ".join(x))

#*************************text stats*************************

df['Readability_Index'] = df['text2'].apply(lambda x: textstat.automated_readability_index(x))
df['Reading_Time'] = df['text2'].apply(lambda x: textstat.reading_time(x))

#*************************Extract first word of text*************************

df['word'] = df['text2'].str.split(' ').str[0]

#*************************Parts of Speech Tagging*************************

df['full_tag'] = df['text2'].apply(lambda x: TextBlob(x).tags)

#*************************Parts of Speech Tagging2*************************


from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV, getAllLemmas

df['tag2'] = df['word'].apply(lambda x: getAllLemmas(x))


#******Categorical Label to Numerica Label

from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
df["label_code"] = ord_enc.fit_transform(df[["label"]])


#*************************filter readability index*************************

df = df[df['Readability_Index'] > -9] 

#************************* Conditional lable to indentify model*************************

#***********************Vector 2****************************
df['Condition_1'] = np.where((df['word']=='if') | (df['word']=='after') | (df['word']=='before') |(df['word']=='when') | (df['word']=='prior') | (df['word']=='during') | (df['word']=='until') |(df['word']=='while') | (df['word']=='following') | (df['word']=='every') | (df['word']=='verify') |(df['word']=='observe') | (df['word']=='check') | (df['word']==''),"v2", "v1")
df['Condition_1_labels'] = np.where((df['label']=='Decide (What)') | (df['label']=='Trigger (What)') | (df['label']=='Verification (What)') | (df['label']=='Waiting (What)'),"Conditional label","No condition")
df['Eliminate'] = np.where((df['Condition_1'] =='v2') & (df['Condition_1_labels']=='No condition'),"Yes", "No")
df = df[df['Eliminate'] != "Yes" ]
df['dic_len']  = df['tag2'].apply(len)
df['Feature3'] =  df['tag2'].apply(lambda x: str(x))



#***********************Vector 3****************************
Verb_1 = 'VERB'
Noun_1= 'NOUN'


df['Condition_2'] = df['Feature3'].str.findall(Verb_1 or (Noun_1 and Verb_1) , flags = re.IGNORECASE)
df['Condition_2'] = df['Condition_2'].astype('string')
df['Condition_2']= df['Condition_2'].astype(str).str.replace(r'\[|\]|', '')
df['Condition_2']= df['Condition_2'].astype(str).str.replace(r'\'|', '')
df['Condition_3'] = np.where((df['Condition_2'] == "VERB") & (df['label']=='Action (How)'),df['word'],0)
                             
                             
                             
#*************************Frecuencies/Lengths/Statistics*************************

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


#*************************BERT Word Vectors*************************

#nlp = spacy_sentence_bert.load_model('en_stsb_bert_large')
#nlp = spacy_sentence_bert.load_model('en_stsb_distilbert_base')

nlp = spacy_sentence_bert.load_model('en_stsb_roberta_large')

df['vector'] = df['text2'].apply(lambda x: nlp(x).vector)

df['vector2'] = df['word'].apply(lambda x: nlp(x).vector)

df['vector3'] = df['Condition_3'].apply(lambda x: nlp(x).vector)


#*************************Final Vector Model*************************

df['Final_vector']  = np.where((df['Condition_1'] == "v2"), df["vector2"], df["vector"])

df['Final_vector']  = np.where((df['Condition_3'] == "Action how verb"), df['vector3'], df['Final_vector'])




#********************Save clean file*************************

df.to_excel (r'C:\Users\EstebanEchandi\Desktop\SOPsv1Clean.xlsx', index = False, header=True)

df_embeddings.to_excel (r'C:\Users\EstebanEchandi\Desktop\Vector3d.xlsx', index = False, header=True)
#**************Dataset split***************
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Final_vector'].tolist(), df['label'].tolist(), test_size=0.3, random_state=690)



#*****************************LABEL***********************************
# train your choice of machine learning classifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)

#--------------------------Confusion matrix---------------------------------
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

#--------------------------ROC--------------------------



#--------------------------Test Algorithm--------------------------

Examples = ['the takeoff must becontinued', 'Action how verb' ,'flight crew']

label = ['Action (What)', 'Action (How)' ,'Actor']

for Examples, label in zip(Examples, label):
  print(Examples,)
  print(f"True Label: {label}, Predicted Label: {clf.predict(nlp(Examples).vector.reshape(1, -1))[0]} \n")



#--------------------------LABEL--------------------------
# train your choice of machine learning classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=10, max_depth=9, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)


#--------------------------Random Forest Visual---------------------------------
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fig = plt.figure(figsize=(100, 90))
plot_tree(clf.estimators_[1], 
                   filled=True, impurity=True, 
          rounded=True)













#Dont use------------------------------PCA Vector Dimensionality reduction-----------------------------

from sklearn.manifold import TSNE

X = list(df["Final_vector"])
#X_embedded = TSNE(n_components=2).fit_transform(X) #2 dimension vector
X_embedded = TSNE(n_components=3).fit_transform(X) #3 dimension vector
df_embeddings = pd.DataFrame(X_embedded)
#df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})  #2 dimension vector
df_embeddings = df_embeddings.rename(columns={0:'x',1:'y',2:'z'}) #3 dimension vector
#df = pd.concat([df, df_embeddings], axis=1).reindex(df.index)

df[['x', 'y', 'z']] = df_embeddings[['x', 'y', 'z']].values

df = df.drop(columns=['x', 'y', 'z'])
df = df.dropna()

















#Dont use--------------------------Visualizations--------------------------

from sklearn.manifold import TSNE

X = list(df["vector"])

#X_embedded = TSNE(n_components=2).fit_transform(X) #2 dimension vector
X_embedded = TSNE(n_components=3).fit_transform(X) #3 dimension vector

df_embeddings = pd.DataFrame(X_embedded)
#df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})  #2 dimension vector
df_embeddings = df_embeddings.rename(columns={0:'x',1:'y',2:'z'}) #3 dimension vector

df2 = pd.concat([df, df_embeddings], axis=1).reindex(df.index)

df2.to_excel (r'C:\Users\EstebanEchandi\Desktop\SOPsv1Clean.xlsx', index = False, header=True)

groups = df2.groupby("label")

for name, group in groups:
     #plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name,markersize=3)
     plt.plot(group["x"], group["y"], group["z"], marker="o", linestyle="", label=name,markersize=3)
plt.legend(loc='best', bbox_to_anchor=(0.5,-0.1))


groups2 = df2.groupby("label2")

for name, group in groups2:
    #plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name,markersize=3)
    plt.plot(group["x"], group["y"], group["z"], marker="o", linestyle="", label=name,markersize=3)
plt.legend(loc='best', bbox_to_anchor=(0.5,-0.1))















#Dont use--------------------------3D Visualizations--------------------------

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = Axes3D(fig)

y = group['y']
x = group['x']
z = group['z']
c = group['label2']

ax.scatter3D(x,y,z, cmap='coolwarm')
# setting title and labels
ax.set_title("3D plot")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)

y = group['y']
x = group['x']
z = group['z']
c = group['label']

ax.scatter3D(x,y,z, cmap='coolwarm')
# setting title and labels
ax.set_title("3D plot")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()

