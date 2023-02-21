#Annika Vetter
#ASTRO 9506S 
#Project Code
#Feb 22nd 2023

############## Import all necessary packages ############################

from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE

import sklearn.preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

###################### define function for SVC, since we'll use it more than once ##########################

def svc(train_data, train_labels, test_data, test_labels):

    '''takes in data from a test/train split, predicts new labels using SVC,
    returns classification report and confursion matrix'''
    
    svc = svm.SVC(kernel='linear',verbose=True)
    clf = svc.fit(train_data, train_labels)

    #predict using svc
    predicted_labels = svc.predict(test_data)

    #rounding predictions to make sure they're either 0 or 1
    predicted_labels = np.round(predicted_labels)

    #Show the classification report
    class_report = print(classification_report(test_labels, predicted_labels))

    #confusion matrix
    cm = confusion_matrix(test_labels, predicted_labels, labels=svc.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svc.classes_)
    cm_plot = disp.plot()
    
    return(class_report, cm_plot)

plt.show()

################## obtain dataset ###########################

#read in known CBEs from Vioque 2018

cbe = Table.read(r'C:\Users\Annika\Documents\Astro Machine Learning\project\cbe.fit')
cbe_df = cbe.to_pandas()

#check how many entires we have (should be 693)
print(cbe_df.shape)


#assign the known CBEs a class colu,m
cbe_df.insert(0, 'class', 0) #0 for known cbes 


#save known CBEs to csv (only needs to run once, just for ease of looking at the data)
#cbe_df.to_csv(r'C:\Users\Annika\Documents\Astro Machine Learning\project\cbe.csv', index=False)


#read in compiled Gaia dr2, 2mass, iphas, etc also from Vioque 2018

gaia = Table.read(r'C:\Users\Annika\Documents\Astro Machine Learning\project\all_sources.fit')
gaia_df = gaia.to_pandas()

#this dataset has over 4 million entries, we'll take a sample of 100 000 to work with 
gaia_df_frac = gaia_df.sample(frac = 0.025)


#assign unknown stars a class column
gaia_df_frac.insert(0, 'class', 1) #1 for unkown sources

#pull only the colour columns from each file
cbe_colours = cbe_df[['class','Gmag','BPmag','RPmag','rmag','Hamag','Jmag','Hmag','Ksmag','W1mag','W2mag','W3mag',"W4mag"]]
gaia_colours = gaia_df_frac[['class','Gmag','BPmag','RPmag','rmag','Hamag','Jmag','Hmag','Ksmag','W1mag','W2mag','W3mag',"W4mag"]]

#combine into one dataset
all_colours = pd.concat([cbe_colours, gaia_colours])


#shuffle the newly combined datset
all_colours = all_colours.sample(frac=1)

#check size of combined dataset
print(all_colours.shape)

################ balancing the data #########################

#oversample minority class (in this case the 693 CBEs)
smote = SMOTE(sampling_strategy = 'minority')
ros = RandomOverSampler(random_state=0)


x_resample, x_resample_labels = SMOTE().fit_resample(all_colours, all_colours['class'])


x_resample['class'] = x_resample_labels

#shuffle new data
x_resample = x_resample.sample(frac=1)

#check size of resampled dataset (see how many new points were added)
print(x_resample.shape)
#print(x_resample)

###################### lets run some tests before combining colours #######################

#train test split of 80/20
train_colours, test_colours = train_test_split(x_resample, test_size=0.2)

#get training data and labels
train_data = train_colours[['Gmag','BPmag','RPmag','rmag','Hamag','Jmag','Hmag','Ksmag','W1mag','W2mag','W3mag',"W4mag"]]
train_labels = train_colours['class']

#get testing data and labels
test_data = test_colours[['Gmag','BPmag','RPmag','rmag','Hamag','Jmag','Hmag','Ksmag','W1mag','W2mag','W3mag',"W4mag"]]
test_labels = test_colours['class']

##################### gradient boosting ####################


# #fit gradient boosting to train data
gbr = GradientBoostingRegressor(verbose = True)
gbr.fit(train_data, train_labels)

# Predict the labels of the test data
predicted_labels = gbr.predict(test_data)

#rounding predictions to make sure its either 0 or 1
predicted_labels = np.round(predicted_labels)

#classification report to see how well it did 
print(classification_report(test_labels, predicted_labels))

#Gradient Boosting did pretty good 

############### random forest ########################

rfr = RandomForestRegressor(verbose = True)
rfr.fit(train_data, train_labels)

#predict using random forest
predicted_labels = rfr.predict(test_data)

#rounding predictions to make sure its either 0 or 1
predicted_labels = np.round(predicted_labels)

#classification report to see how well it performs 
print("Random forest")
print(classification_report(test_labels, predicted_labels))

#Random Forest also pretty good

###################### SVC #######################

#run SVC and generate confusion matrix for just the original columns
svc(train_data, train_labels, test_data, test_labels)

#SVC seems to be performing the best, so we'll stick with this method

################ calculate the colour differences ###################

#we only want the combinations of these columns
no_class = x_resample[['Gmag','BPmag','RPmag','Jmag','Hmag','Ksmag','W1mag','W2mag','W3mag',"W4mag"]]

#find all combinations of columns
combinations = list(itertools.combinations(no_class.columns, 2))

combined_colours = pd.DataFrame()

#for loop for subtracting each combination
for a, b in combinations:
        combined_colours[f'{a}-{b}'] = no_class[a] - no_class[b]
        
#add back in class and r and Halpha columns we excluded from combinations
combined_colours['class'] = x_resample['class']
combined_colours['rmag'] = x_resample['rmag']
combined_colours['Hamag'] = x_resample['Hamag']

#now add an  r - Halpha column, we want this difference but not the combination with the rest of the columns
#so we do it separately here

combined_colours['r-Hamag'] = x_resample['rmag'] - x_resample['Hamag']

pd.set_option('display.max_columns', None)
# check that we have the right number of columns after combinations - should have 45 +4 for a total of 49
print(combined_colours.shape)


#train test split for combinations 

train_colours, test_colours = train_test_split(combined_colours, test_size=0.2)

#get training data and labels
train_data = train_colours[['Gmag-BPmag','Gmag-RPmag','Gmag-Jmag','Gmag-Hmag','Gmag-Ksmag','Gmag-W1mag','Gmag-W2mag',
                           'Gmag-W3mag','Gmag-W4mag','BPmag-RPmag','BPmag-Jmag','BPmag-Hmag','BPmag-Ksmag','BPmag-W1mag',
                           'BPmag-W2mag','BPmag-W3mag','BPmag-W4mag','RPmag-Jmag','RPmag-Hmag','RPmag-Ksmag','RPmag-W1mag',
                           'RPmag-W2mag','RPmag-W3mag','RPmag-W4mag','Jmag-Hmag','Jmag-Ksmag','Jmag-W1mag','Jmag-W2mag',
                           'Jmag-W3mag','Jmag-W4mag','Hmag-Ksmag','Hmag-W1mag','Hmag-W2mag','Hmag-W3mag','Hmag-W4mag',
                           'Ksmag-W1mag','Ksmag-W2mag','Ksmag-W3mag','Ksmag-W4mag','W1mag-W2mag','W1mag-W3mag',
                           'W1mag-W4mag','W2mag-W3mag','W2mag-W4mag','W3mag-W4mag','r-Hamag']]
train_labels = train_colours['class']

#get testing data and labels 
test_data = test_colours[['Gmag-BPmag','Gmag-RPmag','Gmag-Jmag','Gmag-Hmag','Gmag-Ksmag','Gmag-W1mag','Gmag-W2mag',
                           'Gmag-W3mag','Gmag-W4mag','BPmag-RPmag','BPmag-Jmag','BPmag-Hmag','BPmag-Ksmag','BPmag-W1mag',
                           'BPmag-W2mag','BPmag-W3mag','BPmag-W4mag','RPmag-Jmag','RPmag-Hmag','RPmag-Ksmag','RPmag-W1mag',
                           'RPmag-W2mag','RPmag-W3mag','RPmag-W4mag','Jmag-Hmag','Jmag-Ksmag','Jmag-W1mag','Jmag-W2mag',
                           'Jmag-W3mag','Jmag-W4mag','Hmag-Ksmag','Hmag-W1mag','Hmag-W2mag','Hmag-W3mag','Hmag-W4mag',
                           'Ksmag-W1mag','Ksmag-W2mag','Ksmag-W3mag','Ksmag-W4mag','W1mag-W2mag','W1mag-W3mag',
                           'W1mag-W4mag','W2mag-W3mag','W2mag-W4mag','W3mag-W4mag','r-Hamag']]
test_labels = test_colours['class']

# run SVC and generate confusion matrix for all colour combinations
svc(train_data, train_labels, test_data, test_labels)

#this takes quite a while, so lets try some PCA

########################## PCA ##############################

pca = PCA(n_components=49) #49 columns total in our data

#fit PCA to data
pca.fit(combined_colours) 

print("Explained variance ratio: ")
print(pca.explained_variance_ratio_)

#looking for a 95% threshold, and also a 99% threshold to compare to (Vioque, 2018)
print('first 2 components:',np.sum(pca.explained_variance_ratio_[0:1])) #95%
print('first 10 components:',np.sum(pca.explained_variance_ratio_[0:9])) #99%

#generate a scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.bar(PC_values, pca.explained_variance_ratio_, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

plt.show()

X_pca = pca.fit_transform(combined_colours)

#train test split only on the first two components of our PCA fit
x_test, x_train, y_test,y_train = train_test_split(X_pca[:, :1], combined_colours['class'], test_size=0.2)

#run SVC and generate confusion matrix for first two components
svc(x_train, y_train, x_test, y_test)

#train test split only on the first ten components of our PCA fit
x_test, x_train, y_test,y_train = train_test_split(X_pca[:, :9], combined_colours['class'], test_size=0.2)


#run SVC and generate confusion matrix for first ten components
svc(x_train, y_train, x_test, y_test)


#this gives us the "best of both worlds" - high accuracy and fast runtime




