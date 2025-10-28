#-------------------------------------------------------------------------------
# AR_Classifier.py
#
# Main code for the SVM classifier.
#
#    - Edit the lines under ## User Definitions to specify paths and other 
#      parameters.
#    - Outputs three csv feature files with train, test, and validation data 
#      (i.e., the magnetic complexity features, filename, and label); a txt file 
#      weight file used for equalization of features, a txt file performance" 
#      wih classifier statistics, and a pickle file model with the trained model.
#     - Relies on FeaturesetTools.py.
#     - Requires the feature file output by Build_Featureset.py and the data
#       splits (lists of test and val active regions) available on Dryad 
#       (details below)
#       - The feature file for the preconfigured reduced resolution dataset 
#         Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv is available on Dryad 
#         at <insert link here> and for the full resolution dataset 
#         Lat60_Lon60_Nans0_C1.0_24hr_features.csv is available on Dryad at 
#         <insert link here>. It is recommended that you save the feature file
#         in the same classifier_SVM/ directory (i.e., the same directory as the 
#         SVM code), although subsequent code will allow you to specify the path 
#         to those files.
#       - The data splits (lists of test and val active regions) 
#         List_of_AR_in_Test_Data_by_AR.csv, List_of_AR_in_Train_data_by_AR.csv, 
#         and List_of_AR_in_Validation_data_by_AR.csv are available on Dryad 
#         (<insert link here> (reduced resolution) or <insert link here> (full 
#         resolution). It is recommended that you save the data splits files in 
#         the base AR-flares/ directory, although subsequent code will allow you 
#         to specify the path to those files.
#       - Note--if the data splits are not available to the code, the code will 
#         randomly select 10% of active regions for the test and val sets; this 
#         will not result in the same split as the files available on Dryad.
#
# References:
# [1] L. E. Boucheron, T. Vincent, J. A. Grajeda, and E. Wuest, "Solar Active 
#     Region Magnetogram Image Dataset for Studies of Space Weather," arXiv 
#     preprint arXiv:2305.09492, 2023.
#
# Copyright 2022 Laura Boucheron, Jeremy Grajeda, Ellery Wuest
# This file is part of AR-flares
# 
# AR-flares is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
#
# AR-flares is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR 
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with 
# AR-flares. If not, see <https://www.gnu.org/licenses/>. 

# Import Libraries and Tools
import os
import csv
import FeaturesetTools
import numpy as np
import sklearn.svm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

# Key Varibles

# ## User Definitions
# Folder structure
DataFolder = './data/' # base folder for data
SVMFeaturesFolder = DataFolder + 'svm_features/'
ResultsFolder = './results/' # base folder for results
ModelsFolder = ResultsFolder + 'models/'
PlotsFolder = ResultsFolder + 'plots/'

# File paths and parameters
class_type = 'linear' # 'linear' or 'rbf' kernel for SVM classifier
suffix = '' # optional suffix for saving different models, leave as '' for no suffix
classFile  = DataFolder + 'Lat60_Lon60_Nans0_C1.0_24hr_png_224'+suffix+'_features.csv'
trainDataName  = SVMFeaturesFolder + 'Train_Data_by_AR'+suffix+'.csv'
testDataName   = SVMFeaturesFolder + 'Test_Data_by_AR'+suffix+'.csv'
valDataName    = SVMFeaturesFolder + 'Validation_data_by_AR'+suffix+'.csv'
testARList = DataFolder + 'List_of_AR_in_Test_Data_by_AR.csv'
valARList  = DataFolder + 'List_of_AR_in_Validation_Data_by_AR.csv'
outfile = ResultsFolder + 'svm_performance.txt'
weightData = SVMFeaturesFolder + 'Weight_Lat60_Lon60_Nans0_C1.0_24hr'+suffix+'.txt'

# Create directories if they don't exist
for directory in [SVMFeaturesFolder, ResultsFolder, ModelsFolder, PlotsFolder]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Check files

# labels_file = "./data/C1.0_24hr_224_png_Labels.txt"

# # Load labels
# labels_df = pd.read_csv(labels_file, header=None, names=["filename", "label"])
# print(labels_df.head())  # Check first few entries

# Generate Files

# Check for files
if not os.path.exists(trainDataName) or not os.path.exists(testDataName) or not os.path.exists(valDataName):
    # Inform User
    print('\n'+os.path.basename(trainDataName),'or',os.path.basename(testDataName),'or',os.path.basename(valDataName), 'not found.')
    print('Resolving Issue')
    
    # Inform User
    print('Creating', os.path.basename(trainDataName),'and',
              os.path.basename(testDataName),'and',
              os.path.basename(valDataName))
    # Create trainData and testData
    FeaturesetTools.createARBasedSets(classFile,trainDataName,testDataName,valDataName,weightData,testARList,valARList)
    
# Load trainData and testData

# Inform User
print('\nLoading', os.path.basename(trainDataName),'and', 
      os.path.basename(testDataName),'and',
      os.path.basename(valDataName))

# Read and prepare trainData
trainData = np.genfromtxt(trainDataName,delimiter=',',dtype=float,usecols=range(29))
trainLabel = np.genfromtxt(trainDataName,delimiter=',',dtype=int,usecols=29)
trainNames = np.genfromtxt(trainDataName,delimiter=',',dtype=str,usecols=30)

# Read and prpare testData
testData = np.genfromtxt(testDataName,delimiter=',',dtype=float,usecols=range(29))
testLabel = np.genfromtxt(testDataName,delimiter=',',dtype=int,usecols=29)
testNames = np.genfromtxt(testDataName,delimiter=',',dtype=str,usecols=30)

# Read and prpare valData
valData = np.genfromtxt(valDataName,delimiter=',',dtype=float,usecols=range(29))
valLabel = np.genfromtxt(valDataName,delimiter=',',dtype=int,usecols=29)
valNames = np.genfromtxt(valDataName,delimiter=',',dtype=str,usecols=30)

print('Training and evaluating different LinearSVC configurations')

# Define different configurations to try
configs = [
    {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 2000, 'dual': True, 'name': 'Default (C=1)'},
    {'C': 0.1, 'class_weight': 'balanced', 'max_iter': 2000, 'dual': True, 'name': 'Lower C=0.1'},
    {'C': 3.0, 'class_weight': 'balanced', 'max_iter': 2000, 'dual': True, 'name': 'Higher C=3'},
    {'C': 1.0, 'class_weight': None, 'max_iter': 2000, 'dual': True, 'name': 'No class weights'},
    {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 2000, 'dual': False, 'name': 'Primal optimization'}
]

# Colors for different configurations
colors = ['darkorange', 'green', 'blue', 'red', 'purple']

plt.figure(figsize=(12, 8))
best_auc = 0
best_config = None
best_classifier = None

# # Scale the data for better performance
# print("Scaling data...")
# scaler = StandardScaler()
# trainData_scaled = scaler.fit_transform(trainData)
# testData_scaled = scaler.transform(testData)

for config, color in zip(configs, colors):
    print(f"\nTraining LinearSVC with {config['name']}...")
    clf = sklearn.svm.LinearSVC(
        C=config['C'],
        class_weight=config['class_weight'],
        max_iter=config['max_iter'],
        dual=config['dual']
    )
    
    # Train the classifier
    clf.fit(trainData, trainLabel)
    
    # Calculate ROC curve
    y_score = clf.decision_function(testData)
    fpr, tpr, _ = roc_curve(testLabel, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color=color, lw=2, 
             label=f"{config['name']} (AUC = {roc_auc:.3f})")
    
    # Track best configuration
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_config = config
        best_classifier = clf

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different LinearSVC Configurations')
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, alpha=0.3)
plt.savefig(PlotsFolder + 'linear_svm_configs_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nBest configuration: {best_config['name']} with AUC = {best_auc:.3f}")

print('Saving best model')
modelfile = ModelsFolder + 'best_linear_svm_model.pickle'
joblib.dump(best_classifier, modelfile)

# Use best classifier for feature importance analysis
classifier = best_classifier

# Analyze feature importance
feature_importance = np.abs(classifier.coef_[0])
feature_importance = feature_importance / np.sum(feature_importance)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(29), feature_importance)
plt.title('Feature Importance in Linear SVM')
plt.xlabel('Feature Index')
plt.ylabel('Importance (normalized)')
plt.savefig(PlotsFolder + 'feature_importance.png')
plt.close()

# Print top 5 most important features
top_features = np.argsort(feature_importance)[-5:][::-1]
print("\nTop 5 most important features:")
for idx in top_features:
    print(f"Feature {idx}: {feature_importance[idx]:.4f}")

print('Applying learned classifier to test data')
P = classifier.predict(testData)
C = sklearn.metrics.confusion_matrix(testLabel,P,labels=[1,0])
   
#evaluate various performance metrics
tp = C[0,0]; fn=C[0,1]; fp=C[1,0]; tn=C[1,1];
tpr = float(tp)/(tp+fn)
tnr = float(tn)/(tn+fp)
    
hss = float(2*((tp*tn)-(fn*fp)))/((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))
tss = tpr - (1. - tnr)
    
# Save Results

# Inform User
print('Saving Results')
# Print Results
with open(outfile, 'w+') as f:
    f.write('Test data performance')
    f.write('\nTPR = ')
    f.write(str(tpr))
    f.write('\nTNR = ')
    f.write(str(tnr))
    f.write('\nHSS = ')
    f.write(str(hss))
    f.write('\nTSS = ')
    f.write(str(tss))

print('Applying learned classifier to validation data')
P = classifier.predict(valData)
C = sklearn.metrics.confusion_matrix(valLabel,P,labels=[1,0])
#evaluate various performance metrics
tp = C[0,0]; fn=C[0,1]; fp=C[1,0]; tn=C[1,1];
tpr = float(tp)/(tp+fn)
tnr = float(tn)/(tn+fp)
    
hss = float(2*((tp*tn)-(fn*fp)))/((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))
tss = tpr - (1. - tnr)
    
print('Saving Results')
# Print Results
with open(outfile, 'a+') as f:
    f.write('\nValidation data performance')
    f.write('\nTPR = ')
    f.write(str(tpr))
    f.write('\nTNR = ')
    f.write(str(tnr))
    f.write('\nHSS = ')
    f.write(str(hss))
    f.write('\nTSS = ')
    f.write(str(tss))

print('Applying learned classifier to training data')
P = classifier.predict(trainData)
C = sklearn.metrics.confusion_matrix(trainLabel,P,labels=[1,0])
#evaluate various performance metrics
tp = C[0,0]; fn=C[0,1]; fp=C[1,0]; tn=C[1,1];
tpr = float(tp)/(tp+fn)
tnr = float(tn)/(tn+fp)
    
hss = float(2*((tp*tn)-(fn*fp)))/((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))
tss = tpr - (1. - tnr)
    
print('Saving Results')
# Print Results
with open(outfile, 'a+') as f:
    f.write('\nTraining data performance')
    f.write('\nTPR = ')
    f.write(str(tpr))
    f.write('\nTNR = ')
    f.write(str(tnr))
    f.write('\nHSS = ')
    f.write(str(hss))
    f.write('\nTSS = ')
    f.write(str(tss))
    
# Inform user
print('Process Complete')
