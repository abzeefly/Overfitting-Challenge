import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from mlxtend.classifier import StackingClassifier

#######################################################################################################################

#Name : Abdullah Zaid Ansari
#Student ID : z5099229
#Machine Learning Project

########################################################################################################################


#Import the test and train files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
labels = train.columns.drop(['id', 'target'])

#save data into variables and remove unused features
X = train.drop(['id','target'],axis = 1)
Y = train['target']
X_eval = test.drop(['id'], axis = 1)

#Get the most used features using rfe
modelXGB = XGBClassifier(max_depth = 2, gamma = 2, eta = 0.8, reg_alpha = 0.5, reg_lambda = 0.5)
rfe = RFE(modelXGB, n_features_to_select = 5)
rfe.fit(X,Y)

#transform the test and train sets with changed features
X_fs = rfe.transform(X)
X_fs_eval = rfe.transform(X_eval)

#balances out the training wrt to targets using SMOTE
smote = SMOTE(sampling_strategy='minority', n_jobs=-1)
X_pd, y_pd = smote.fit_resample(X_fs,Y)
new_table = pd.DataFrame(X_pd)
new_table['target'] = y_pd

#Save the normalized features and target to variables
normX = new_table.drop(['target'], axis = 1)
normY = new_table['target']

#Import the models to be used for this task
modelLR = LogisticRegression(solver = 'liblinear',C = 0.05, penalty = 'l2', class_weight ='balanced', max_iter = 10)
modelDT = DecisionTreeClassifier(random_state = 0, max_depth = 3, min_samples_leaf = 5, min_samples_split = 2 )
modelXGB = XGBClassifier(max_depth = 2, gamma = 2, eta = 0.8, reg_alpha = 0.5, reg_lambda = 0.5)

#turn these datasets to scalers for consistent fitting
scaler = StandardScaler()
normX = scaler.fit_transform(normX)
X_eval = scaler.fit_transform(X_fs_eval)

#stack the classifiers using mlxtend, make LR model the meta_classifier to give it more weight
m = StackingClassifier(
    classifiers=[
        modelLR,
        modelDT,
        modelXGB
    ],
    use_probas=True,
    meta_classifier= modelLR
)

#fit the model and save the predictions
m.fit(normX, normY)
pred = m.predict_proba(X_fs_eval)[:,1]

#save the results into the file
submission = pd.read_csv('sample_submission.csv')
submission['target'] = pred
submission.to_csv('sample_submission.csv', index = False)

