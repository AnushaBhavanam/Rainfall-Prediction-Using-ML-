import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

rainfall_data=pd.read_csv("weatherAUS.csv")
rainfall_data.shape
rainfall_data.info()
rainfall_data.isna().sum()
rainfall_data.drop('Date',inplace=True,axis=1)
rainfall_data.drop('Location',inplace=True,axis=1)
rainfall_data['RainToday'].replace({'No':0,'Yes':1},inplace=True)
rainfall_data['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)

#Handling Class Imbalance
fig = plt.figure(figsize=(8,5))
rainfall_data.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color=['m','green'],rot=0,alpha=0.9)
plt.title('RainTomorrow Indicator No(0) and Yes(1) in the Imbalanced Dataset')
from sklearn.utils import resample
# stores all the '0' values in it
no = rainfall_data[rainfall_data.RainTomorrow == 0]
# stores all the '1' values in it
yes = rainfall_data[rainfall_data.RainTomorrow == 1]

# as we are oversampling the minority class i.e 'yes' values by using the resample()

yes_oversample = resample(yes, replace=True , n_samples=len(no) , random_state = 123)
oversample = pd.concat([yes_oversample , no])

# Now let us plot the figure

fig=plt.figure(figsize =(8,5))
oversample.RainTomorrow.value_counts(normalize=True).plot(kind='bar',color=['m','green'],rot=0,alpha=0.9)
plt.title('RainTomorrow Indicator No(0) and Yes(1) after oversampling (Balanced Dataset)')
plt.show()

# Missing data
sns.heatmap(oversample.isna(),cbar=False, cmap='PuBu')
total = oversample.isnull().sum().sort_values(ascending=False)
percent = (oversample.isnull().sum()/oversample.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing.head(23 )

# IMPUTATION and TRANSFORMATION

oversample.select_dtypes(include=['object']).columns
# Impute the categorical variable with 'Mode'

oversample['WindGustDir'] = oversample['WindGustDir'].fillna(oversample['WindGustDir'].mode()[0])
oversample['WindDir9am'] = oversample['WindDir9am'].fillna(oversample['WindDir9am'].mode()[0])
oversample['WindDir3pm'] = oversample['WindDir3pm'].fillna(oversample['WindDir3pm'].mode()[0])
oversample.isna().sum()

#We will convert the categorical features to continuous features with label encoding

from sklearn.preprocessing import LabelEncoder
lencoders = {}
for col in oversample.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    oversample[col] = lencoders[col].fit_transform(oversample[col])

import warnings
warnings.filterwarnings("ignore")

#Multiple Imputation by Chained Equations

from sklearn.impute import IterativeImputer
MiceImputed = oversample.copy(deep=True)
mice_imputer = IterativeImputer()
MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversample)
MiceImputed.isna().sum()

# Outliers

Q1 = MiceImputed.quantile(0.25)
Q3 = MiceImputed.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
## Removing the outliers from the dataset

MiceImputed = MiceImputed[~((MiceImputed<(Q1-1.5*IQR))|(MiceImputed>(Q3+1.5*IQR))).any(axis=1)]
MiceImputed.shape

# Correlation

corr = MiceImputed.corr()
f,ax = plt.subplots(figsize=(20,20))
mask = np.triu(np.ones_like(corr,dtype = np.bool))
cmap = sns.diverging_palette(650,300,as_cmap=True)
sns.heatmap(corr,mask=mask,cmap=cmap,vmax = None,center = 0,square = True,annot = True,linewidth = .5,cbar_kws = {'shrink':.9})
# Standardizing data
# initialising the minmax scaler function in to r_scaler
# scaling the dataset keeping the columns name

from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
#r_scaler.fit(MiceImputed)
modified_data = pd.DataFrame(r_scaler.fit_transform(MiceImputed),index=MiceImputed.index,columns=MiceImputed.columns)
# Feature Importance using Filter Method(Chi-Square)

from sklearn.feature_selection import SelectKBest , chi2
X = modified_data.loc[:,modified_data.columns!='RainTomorrow']
y = modified_data[['RainTomorrow']]
selector = SelectKBest(chi2,k=10)
selector.fit(X,y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)])
features = MiceImputed[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']]
target = MiceImputed['RainTomorrow']

# Split into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=12345)

# Normalize Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, roc_curve, classification_report
def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))
    
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plot_roc_cur(fper, tper)
    
    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')
    
    return model, accuracy, roc_auc, coh_kap, time_taken

# Random Forest
from sklearn.ensemble import RandomForestClassifier

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

model_rf = RandomForestClassifier(**params_rf)
model_rf, accuracy_rf, roc_auc_rf, coh_kap_rf, tt_rf = run_model(model_rf, X_train, y_train, X_test, y_test)
pickle.dump(model_rf, open('rainfall.pkl','wb'))





