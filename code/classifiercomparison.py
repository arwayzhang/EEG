
from __future__ import print_function


import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder, binarize
import matplotlib.pyplot as plt
import itertools 
print(__doc__)

import numpy as np

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN # use "from dbn.tensorflow import SupervisedDBNClassification" for computations on TensorFlow
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, learning_curve, ShuffleSplit
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from time import time, strftime, localtime
import math


def Confusion_matrix_plot2(clf_name, prediction, y_test, classes = ['1', '2', '3','4','R','W'], cmap=plt.get_cmap('Blues')):
    plt.figure(figsize=(8,8))
    cm = confusion_matrix(y_test, prediction)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)
    ct = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > ct else "black", fontsize=18)
    plt.title('Confusion matrix of {:}'.format(clf_name), fontsize=14)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.savefig((confufilename + "_{:}.png").format(clf_name), dpi=300)


def Confusion_matrix_plot(clf_name, clf, X_test, y_test, classes = ['1', '2', '3','4','R','W'], cmap=plt.get_cmap('Blues')):
    plt.figure(figsize=(8,8))
    cm = confusion_matrix(y_test, clf.predict(X_test))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)
    ct = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > ct else "black", fontsize=18)
    plt.title('Confusion matrix of {:}'.format(clf_name), fontsize=14)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.savefig((confufilename + "_{:}.png").format(clf_name), dpi=300)

################################################33

####define filename#################################################################################################################################################


numstring = "1"

##################

confufilename = "Confusion_Matrix" + str(numstring)


traindatafilename = 'train'+numstring+'.csv'
trainlabeldatafilename = 'label_train'+numstring+'.csv'

###############################################################################

def tuneLR(X_train, y_train, X_test, y_test, n_jobs=4):
	clfname='LogisticRegression'
	param_grid = [{'C': [1e3], 'penalty': ['l1','l2']}]
	clf = GridSearchCV(LogisticRegression(), param_grid, cv=5)
	_ = clf.fit(X_train, y_train)

	bestparam = clf.best_params_

	print(bestparam)
	
	clf = LogisticRegression(**clf.best_params_)

	start = time()
	_ = clf.fit(X_train, y_train)
	print('Fitting {:} finished at {:} with runtime of {:.2f} secs.'.format(clfname, strftime("%H:%M:%S", localtime()), time()-start))
	print(classification_report(Y_test, clf.predict(X_test)))

	Confusion_matrix_plot(clfname, clf, X_test, y_test)



def tuneGNB(X_train, y_train, X_test, y_test, n_jobs=4):
	clfname='Gaussian_Naive_Bayes'
	param_grid = [{'priors': [None]}]
	clf = GridSearchCV(GaussianNB(), param_grid, cv=10)
	_ = clf.fit(X_train, y_train)

	bestparam = clf.best_params_
	clf = GaussianNB(**clf.best_params_)

	start = time()
	_ = clf.fit(X_train, y_train)
	print('Fitting {:} finished at {:} with runtime of {:.2f} secs.'.format(clfname, strftime("%H:%M:%S", localtime()), time()-start))
	print(classification_report(Y_test, clf.predict(X_test)))

	Confusion_matrix_plot(clfname, clf, X_test, y_test)



# Setting up
def datanize(filename,header=None,firstnoneed=1):
	if firstnoneed==1:
		inputdata = pd.read_csv(filename, header=None, skiprows=[0])
		for col in inputdata.columns:
			if not inputdata[col].empty:
				inputdata[col] = LabelEncoder().fit_transform(inputdata[col])
	else:
		inputdata = pd.read_csv(filename, header=None)
		for col in inputdata.columns:
			if not inputdata[col].empty:
				inputdata[col] = LabelEncoder().fit_transform(inputdata[col])
	return inputdata

## split dataset #####################################################################

dataframe = datanize(traindatafilename,header=None,firstnoneed=0)

X = dataframe.values


X = (X / 10000).astype(np.float32)


labelframe = datanize(trainlabeldatafilename,header=None,firstnoneed=0)

Y = labelframe.values

Y = Y.T[0]


# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

##########################################################################################

### RBM-Logistic Models we will use #######################################################3
logistic = linear_model.LogisticRegression()
dbn = UnsupervisedDBN(hidden_layers_structure=[500, 500],
                      batch_size=32,
                      learning_rate_rbm=0.05,
                      n_epochs_rbm=10,
                      activation_function='relu')

classifier = Pipeline(steps=[('dbn', dbn),
                             ('logistic', logistic)])

classifier.fit(X_train, Y_train)

DBNprediction = classifier.predict(X_test)


###############################################################################
# Evaluation

print("Logistic regression using RBM features:\n%s\n" % (
    classification_report(
        Y_test,
        DBNprediction)))


Confusion_matrix_plot2('RBM-Logistic', DBNprediction, Y_test, classes = ['1', '2', '3','4','R','W'], cmap=plt.get_cmap('Blues'))


##### GNB, LR ###################################################################

tuneGNB(X_train, Y_train, X_test, Y_test, n_jobs=4)
tuneLR(X_train, Y_train, X_test, Y_test, n_jobs=4)

### SVM ######################################################################

clf = svm.LinearSVC()

start = time()

clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)

print('Classification {:} finished at {:} with runtime of {:.2f} secs.'.format('SVM', strftime("%H:%M:%S", localtime()), time()-start))

print("SVM:\n%s\n" % (
    classification_report(
        Y_test,
        prediction)))


Confusion_matrix_plot2('SVM', prediction, Y_test, classes = ['1', '2', '3','4','R','W'], cmap=plt.get_cmap('Blues'))


