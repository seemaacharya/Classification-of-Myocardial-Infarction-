# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 12:01:51 2022

@author: DELL
"""


#1)import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#2)loading dataset
data=pd.read_csv("MIC.csv")
data.columns
data.shape
#1700 rows,124 columns

#3)we will remove the ID column as ID does not play any role in prediction.
data.drop("ID",axis=1,inplace=True)
data.shape
#1700 rows, 123 columns

#4)Missing values
missing_values_list=data.isna().sum().tolist()
missing_values=data.isna().sum()
missing_values_list
data.columns.to_list()


#5)continuous_missingvalues_list
continuous_missing_list=[]
continuous_values=["AGE","S_AD_KBRIG","D_AD_KBRIG","S_AD_ORIT","D_AD_ORIT","K_BLOOD",
                   "NA_BLOOD","ALT_BLOOD","AST_BLOOD","KFK_BLOOD","L_BLOOD","ROE"]

for i in continuous_values:
    a=data[i].isna().sum()
    continuous_missing_list.append(a)
    
len(continuous_values)
#12 columns in continuous_values where missing values are there.
    

#6)categorical_missing_list
categorical_missing_list=[]
categorical_values=["SEX",
                    "INF_ANAM",
                    "STENOK_AN",
                    "FK_STENOK",
                    "IBS_POST",
                    "IBS_NASL",
                    "GB",
                    "SIM_GIPERT",
                    "DLIT_AG",
                    "ZSN_A",
                    "nr_11",
                    "nr_01",
                    "nr_02",
                    "nr_03",
                    "nr_04",
                    "nr_07",
                    "nr_08",
                    "np_01",
                    "np_04",
                    "np_05",
                    "np_07",
                    "np_08",
                    "np_09",
                    "np_10",
                    "endocr_01",
                    "endocr_02",
                    "endocr_03",
                    "zab_leg_01",
                    "zab_leg_02",
                    "zab_leg_03",
                    "zab_leg_04",
                    "zab_leg_06",
                    "O_L_POST",
                    "K_SH_POST",
                    "MP_TP_POST",	
                    "SVT_POST",
                    "GT_POST",
                    "FIB_G_POST",
                    "ant_im",
                    "lat_im",
                    "inf_im",
                    "post_im",
                    "IM_PG_P",
                    "ritm_ecg_p_01",
                    "ritm_ecg_p_02",
                    "ritm_ecg_p_04",
                    "ritm_ecg_p_06",	
                    "ritm_ecg_p_07",
                    "ritm_ecg_p_08",
                    "n_r_ecg_p_01",
                    "n_r_ecg_p_02",
                    "n_r_ecg_p_03",
                    "n_r_ecg_p_04",
                    "n_r_ecg_p_05",
                    "n_r_ecg_p_06",
                    "n_r_ecg_p_08",
                    "n_r_ecg_p_09",
                    "n_r_ecg_p_10",
                    "n_p_ecg_p_01",
                    "n_p_ecg_p_03",
                    "n_p_ecg_p_04",
                    "n_p_ecg_p_05",
                    "n_p_ecg_p_06",
                    "n_p_ecg_p_07",
                    "n_p_ecg_p_08",
                    "n_p_ecg_p_09",
                    "n_p_ecg_p_10",
                    "n_p_ecg_p_11",
                    "n_p_ecg_p_12",
                    "fibr_ter_01",
                    "fibr_ter_02",
                    "fibr_ter_03",
                    "fibr_ter_05",
                    "fibr_ter_06",
                    "fibr_ter_07",
                    "fibr_ter_08",
                    "GIPO_K",
                    "GIPER_NA",
                    "TIME_B_S",
                    "R_AB_1_n",
                    "R_AB_2_n",
                    "R_AB_3_n",
                    "NA_KB",
                    "NOT_NA_KB",
                    "LID_KB",
                    "NITR_S",
                    "NA_R_1_n",
                    "NA_R_2_n",
                    "NA_R_3_n",
                    "NOT_NA_1_n",
                    "NOT_NA_2_n",
                    "NOT_NA_3_n",
                    "LID_S_n",
                    "B_BLOK_S_n",
                    "ANT_CA_S_n",
                    "GEPAR_S_n",
                    "ASP_S_n",
                    "TIKL_S_n",
                    "TRENT_S_n",	
                    "FIBR_PREDS", 
                    "PREDS_TAH",
                    "JELUD_TAH",
                    "FIBR_JELUD",
                    "A_V_BLOK",
                    "OTEK_LANC",
                    "RAZRIV",
                    "DRESSLER",
                    "ZSN",
                    "REC_IM",
                    "P_IM_STEN", 
                    "LET_IS"]
for i in categorical_values:
    c=data[i].isna().sum()
    categorical_missing_list.append(c)

len(categorical_values)
#111

#7)Median imputation for numerical data(median imputation is done when numerical data is skewed)
#checking the skewness of the numerical data so that we can do the median impuation
plt.hist(data.AGE)#left skewed
plt.hist(data.S_AD_KBRIG)#left skewed
plt.hist(data.D_AD_KBRIG)#left skewed
plt.hist(data.S_AD_ORIT)#left skewed
plt.hist(data.K_BLOOD)#right skewed 
plt.hist(data.NA_BLOOD)#right
plt.hist(data.ALT_BLOOD)#right
plt.hist(data.AST_BLOOD) #right 
plt.hist(data.KFK_BLOOD) #uneven distributed   
plt.hist(data.L_BLOOD) #right  
plt.hist(data.ROE)#right

#median imputation
#creating a list of continuous missing values

continuous_missing_values=["AGE","S_AD_KBRIG","D_AD_KBRIG","S_AD_ORIT","D_AD_ORIT","K_BLOOD",
                   "NA_BLOOD","ALT_BLOOD","AST_BLOOD","KFK_BLOOD","L_BLOOD","ROE"]

for column in continuous_missing_values:
    data[column].fillna(data[column].median(),inplace=True)
    
    
data.isna().sum().head(90)   

#mode imputation for categorical data
#creating a list of categorical missing values
categorical_missing_values=["SEX",
                    "INF_ANAM",
                    "STENOK_AN",
                    "FK_STENOK",
                    "IBS_POST",
                    "IBS_NASL",
                    "GB",
                    "SIM_GIPERT",
                    "DLIT_AG",
                    "ZSN_A",
                    "nr_11",
                    "nr_01",
                    "nr_02",
                    "nr_03",
                    "nr_04",
                    "nr_07",
                    "nr_08",
                    "np_01",
                    "np_04",
                    "np_05",
                    "np_07",
                    "np_08",
                    "np_09",
                    "np_10",
                    "endocr_01",
                    "endocr_02",
                    "endocr_03",
                    "zab_leg_01",
                    "zab_leg_02",
                    "zab_leg_03",
                    "zab_leg_04",
                    "zab_leg_06",
                    "O_L_POST",
                    "K_SH_POST",
                    "MP_TP_POST",	
                    "SVT_POST",
                    "GT_POST",
                    "FIB_G_POST",
                    "ant_im",
                    "lat_im",
                    "inf_im",
                    "post_im",
                    "IM_PG_P",
                    "ritm_ecg_p_01",
                    "ritm_ecg_p_02",
                    "ritm_ecg_p_04",
                    "ritm_ecg_p_06",	
                    "ritm_ecg_p_07",
                    "ritm_ecg_p_08",
                    "n_r_ecg_p_01",
                    "n_r_ecg_p_02",
                    "n_r_ecg_p_03",
                    "n_r_ecg_p_04",
                    "n_r_ecg_p_05",
                    "n_r_ecg_p_06",
                    "n_r_ecg_p_08",
                    "n_r_ecg_p_09",
                    "n_r_ecg_p_10",
                    "n_p_ecg_p_01",
                    "n_p_ecg_p_03",
                    "n_p_ecg_p_04",
                    "n_p_ecg_p_05",
                    "n_p_ecg_p_06",
                    "n_p_ecg_p_07",
                    "n_p_ecg_p_08",
                    "n_p_ecg_p_09",
                    "n_p_ecg_p_10",
                    "n_p_ecg_p_11",
                    "n_p_ecg_p_12",
                    "fibr_ter_01",
                    "fibr_ter_02",
                    "fibr_ter_03",
                    "fibr_ter_05",
                    "fibr_ter_06",
                    "fibr_ter_07",
                    "fibr_ter_08",
                    "GIPO_K",
                    "GIPER_NA",
                    "TIME_B_S",
                    "R_AB_1_n",
                    "R_AB_2_n",
                    "R_AB_3_n",
                    "NA_KB",
                    "NOT_NA_KB",
                    "LID_KB",
                    "NITR_S",
                    "NA_R_1_n",
                    "NA_R_2_n",
                    "NA_R_3_n",
                    "NOT_NA_1_n",
                    "NOT_NA_2_n",
                    "NOT_NA_3_n",
                    "LID_S_n",
                    "B_BLOK_S_n",
                    "ANT_CA_S_n",
                    "GEPAR_S_n",
                    "ASP_S_n",
                    "TIKL_S_n",
                    "TRENT_S_n",	
                    "FIBR_PREDS", 
                    "PREDS_TAH",
                    "JELUD_TAH",
                    "FIBR_JELUD",
                    "A_V_BLOK",
                    "OTEK_LANC",
                    "RAZRIV",
                    "DRESSLER",
                    "ZSN",
                    "REC_IM",
                    "P_IM_STEN", 
                    "LET_IS"]    
    
for column in categorical_missing_values:
    data[column].fillna(data[column].mode()[0],inplace=True)
    
data.isna().sum().head(20)  
data.isna().sum().tail(20)
data.isna().sum().head(90)
#all the missing values are fixed now.

data_new=data["LET_IS"]
data_new.dtypes

#8)Normalization of continuous data
for column in continuous_missing_values:
    data[column]=(data[column]-data[column].min())/(data[column].max()-data[column].min())
data    
    
#9)Feature selection using Chi2 and SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
    
x=data.iloc[:,0:122]
y=data.iloc[:,122]

test=SelectKBest(score_func=chi2,k=4)
fit=test.fit(x,y)
fit.scores_
len(fit.scores_)    
#122

#making the dict of featues and scores to get the idea of weights of each feature

column_list=data.columns.tolist()
column_list.remove("LET_IS")
column_list

d=dict();
for i in range(len(fit.scores_)):
    d[column_list[i]]=fit.scores_[i]
d    
    

#selecting high scores features against features(threshold =15 out of 10,15,20)
high_score_features={k:v for (k,v) in d.items() if v>15}
high_score_features
len(high_score_features)
#There are 53 high score features out of 123 features

#selected_features=
selected_features=[]
for k,v in high_score_features.items():
    selected_features.append(k)
selected_features    

data.columns
#making dataset compatible by keeping only the selected features and dropping the rest of the features
for i in data.columns:
    if i not in selected_features:
        data.drop(i,axis=1,inplace=True)
    else:
        pass

data["LET_IS"]=data_new
data
#checking the dataset is balanced or not
data.LET_IS.value_counts()
#we see that type 0==1429 count and type=5 is showing only 12 counts, meaning our data is highly imbalanced.

#SMOTE
pip install imblearn
from imblearn.over_sampling import SMOTE
smote=SMOTE()
    
#train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)  

#apply smote technique
x_train_smote,y_train_smote=smote.fit_resample(x_train,y_train)
unique,counts=np.unique(y_train_smote,return_counts=True)
print(list(zip(unique,counts)))  
    
#model building
#1)Decision tree
from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier(criterion="entropy",max_depth=10)
model_dt.fit(x_train_smote,y_train_smote)
pred_dt=model_dt.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

y_pred_balanced=model_dt.predict(x_train_smote)
#train accuracy
train_dt_acc=accuracy_score(y_train_smote,y_pred_balanced)
train_dt_acc
#95.78%

#test_accuracy
test_dt_accuracy=accuracy_score(y_test,pred_dt)
test_dt_accuracy
#76.47%

#we see that train accuracy is higher than the test sccuracy, we will fix this by GridSearchCV
from sklearn.model_selection import GridSearchCV
max_depth=np.array(range(1,10))
param_grid=dict(max_depth=max_depth)
model=DecisionTreeClassifier(criterion="entropy")
grid=GridSearchCV(estimator=model,param_grid=param_grid)
grid.fit(x_train_smote,y_train_smote)
print(grid.best_score_)
#92.12%
print(grid.best_params_)
#max_depth=9

#2)KNN
from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier(n_neighbors=2)
model_knn.fit(x_train_smote,y_train_smote)
#train_accuracy
pred_knn_train=model_knn.predict(x_train_smote)
pred_knn_train_accuracy=accuracy_score(y_train_smote,pred_knn_train)
pred_knn_train_accuracy
#100%

#test accuracy
pred_knn_test=model_knn.predict(x_test)
pred_knn_test_accuracy=accuracy_score(y_test,pred_knn_test)
#66.76%

#we see that train accuracy is higher that test accuracy,
#we will fix this by GridSearchCV(by finding the optimum value of n_neighbors for improving the accuracy of KNN model)
from sklearn.model_selection import GridSearchCV
import numpy

n_neighbors=numpy.array(range(1,40))
param_grid=dict(n_neighbors=n_neighbors)
model_1=KNeighborsClassifier()
grid=GridSearchCV(estimator=model_1,param_grid=param_grid)
grid.fit(x_train_smote,y_train_smote)
print(grid.best_score_)
#96.84
print(grid.best_params_)
#n_neighbors=2


#3)Random Forest
from sklearn.ensemble import RandomForestClassifier
num_trees=130
max_features=4
model_RF=RandomForestClassifier(n_estimators=num_trees,max_features=max_features)
model_RF.fit(x_train_smote,y_train_smote)
#train accuracy
y_pred_RF_train=model_RF.predict(x_train_smote)
train_RF_accuracy=accuracy_score(y_train_smote,y_pred_RF_train)
#100%
y_pred_RF_test=model_RF.predict(x_test)
test_RF_accuracy=accuracy_score(y_test,y_pred_RF_test)
#88.52%

#train accuracy is higher than test accuracy

#Finding the optimal value for n_estimators and max_features to improve the accuracy for RF
#by using GridSearchCV
from sklearn.model_selection import GridSearchCV
n_neighbors=np.array(range(100,150))
param_grid=dict(n_estimators=n_neighbors)
model=RandomForestClassifier()
grid=GridSearchCV(estimator=model,param_grid=param_grid)
grid.fit(x_train_smote,y_train_smote)
print(grid.best_score_)
#99.48
print(grid.best_params_)
#{'n_estimators': 110}



from sklearn.model_selection import GridSearchCV
max_features=np.array(range(1,5))
param_grid=dict(max_features=max_features)
model=RandomForestClassifier()
grid=GridSearchCV(estimator=model,param_grid=param_grid)
grid.fit(x_train_smote,y_train_smote)
print(grid.best_score_)
#99.48
print(grid.best_params_)
#{'max_features': 1}


#RF gives the highest accuracy for train and test models. Hence, RF is the best model for prediction ad for classification and is selected for the final deployment.




    
    
    








                    




