#!/usr/bin/env python
# coding: utf-8


import pandas as pd
df = pd.read_csv("C:/Users/ASUS/OneDrive/MineTasks/H_C/Actual_Data.csv")
df.head(10)

df.drop_duplicates()

df.rename(columns = {'Weight':'Weight_kg'}, inplace = True)
df.rename(columns = {'Height':'Height_cm'}, inplace = True)
df.rename(columns = {'Glucose_Level_Before':'Glucose_Level_Before_mg/dL'}, inplace = True)
df.rename(columns = {'Glucose_Level_After':'Glucose_Level_After_mg/dL'}, inplace = True)
df.rename(columns = {'Heart_Rate':'Heart_Rate_bpm'}, inplace = True)

df.isna().sum()

import numpy as np
# Replace infinite values with zero
df['Age'] = df['Age'].fillna(0).replace([np.inf, -np.inf], 0)
df['Weight_kg'] = df['Weight_kg'].fillna(0).replace([np.inf, -np.inf], 0)
df['Height_cm'] = df['Height_cm'].fillna(0).replace([np.inf, -np.inf], 0)
df['SBP'] = df['SBP'].fillna(0).replace([np.inf, -np.inf], 0)
df['DBP'] = df['DBP'].fillna(0).replace([np.inf, -np.inf], 0)
df['Heart_Rate_bpm'] = df['Heart_Rate_bpm'].fillna(0).replace([np.inf, -np.inf], 0)
df['Glucose_Level_Before_mg/dL'] = df['Glucose_Level_Before_mg/dL'].fillna(0).replace([np.inf, -np.inf], 0)
df['Glucose_Level_After_mg/dL'] = df['Glucose_Level_After_mg/dL'].fillna(0).replace([np.inf, -np.inf], 0)

# Convert the column to integers
df['Age'] = df['Age'].astype(int)
df['Weight_kg'] = df['Weight_kg'].astype(int)
df['Height_cm'] = df['Height_cm'].astype(int)
df['SBP'] = df['SBP'].astype(int)
df['DBP'] = df['DBP'].astype(int)
df['Heart_Rate_bpm'] = df['Heart_Rate_bpm'].astype(int)
df['Glucose_Level_Before_mg/dL'] = df['Glucose_Level_Before_mg/dL'].astype(int)
df['Glucose_Level_After_mg/dL'] = df['Glucose_Level_After_mg/dL'].astype(int)

df['Family_History'] = np.where(df['Family_History']=='Yes', 1, 0)
df['Diabetes_Chance_Level'] = np.where(df['Diabetes_Chances']=='Low', 0, np.where(df['Diabetes_Chances']=='Medium', 1,2))
df['Age_group'] = pd.cut(df['Age'], bins = [18,26,36,46, float('inf')], labels = ['18-25','26-35','36-45','45+'])
conditions = [
    (df['SBP']<=120) & (df['DBP']<=80),
    (df['SBP']>120) & (df['SBP']<=139) & (df['DBP']<=80),
    (df['SBP']>139) & (df['DBP']>=80) |
    (df['SBP']<=120) & (df['DBP']>=80)|
    (df['SBP']>=120) & (df['DBP']>=80)
]
values = ['Normal','Elevated','Hyepertension']
df['BP_Group'] = np.select(conditions,values)

conditions = [
    (df['Glucose_Level_Before_mg/dL']<100) & (df['Glucose_Level_After_mg/dL']<140),
    ((df['Glucose_Level_Before_mg/dL']>=100) & (df['Glucose_Level_Before_mg/dL']<=125) | 
     (df['Glucose_Level_After_mg/dL']>=140) & (df['Glucose_Level_After_mg/dL']<=199)),
    (df['Glucose_Level_Before_mg/dL']>=126) | (df['Glucose_Level_After_mg/dL']>=200)
]
values = ['Normal','Prediabetes','Diabete']
df['Glucose_Level'] = np.select(conditions,values)

df['HeartRate_Group'] = pd.cut(df['Heart_Rate_bpm'], bins = [-float('inf'),60,81, float('inf')], labels = ['Low','Normal','High'])

df.head(10)


#df['Height'] = df['Height'] / 100
df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)
df['BMI'] = df['BMI'].round(3)

df['Overweight'] = np.where((df['BMI'] >= 25) & (df['BMI'] < 29.9), 1, 0)
df['Obese'] = np.where(df['BMI']>=30, 1, 0)
df.head(50)


# Define the desired column order
new_order = ['Age','Gender','Weight_kg','Height_cm','SBP','DBP','Family_History','Heart_Rate_bpm', 'Glucose_Level_Before_mg/dL','Glucose_Level_After_mg/dL','BMI','Overweight','Obese','Age_group','BP_Group','HeartRate_Group','Diabetes_Chances','Diabetes_Chance_Level']
# Use the reindex method to reorder the columns
df = df.reindex(columns=new_order)
df.head(10)


df.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#required columns based on heatmap
new_order = ['Age','Weight_kg','Height_cm','SBP','DBP','Family_History','Heart_Rate_bpm', 'Glucose_Level_Before_mg/dL','Glucose_Level_After_mg/dL','BMI','Overweight','Obese','Diabetes_Chance_Level']
# Use the reindex method to reorder the columns
dff = df.reindex(columns=new_order)
dff.head(10)

dff.columns


from sklearn.model_selection import train_test_split as split, GridSearchCV

(train_data, test_data) = split(dff, test_size=0.2)

print(f"Original data has {train_data.shape[0]} train data and {test_data.shape[0]} test data")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Create the random forest classifier
model = RandomForestClassifier()

# Train the model on the training data
model.fit(train_data.drop("Diabetes_Chance_Level", axis=1), train_data["Diabetes_Chance_Level"])

# Predict the labels of the testing data
predictions = model.predict(test_data.drop("Diabetes_Chance_Level", axis=1))

# Calculate the accuracy
accuracy = accuracy_score(predictions, test_data["Diabetes_Chance_Level"])

print("Accuracy:", accuracy)


import pickle

# Creating a pickle file for the classifier
filename = 'C:/Users/ASUS/OneDrive/MineTasks/H_C/diabetes-prediction-rfc-model.pkl'
pickle.dump(model, open(filename, 'wb'))
