import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder, binarize
import matplotlib.pyplot as plt
from time import time, strftime, localtime
import itertools 

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


from dbn.tensorflow import SupervisedDBNClassification

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, learning_curve, ShuffleSplit
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn import linear_model

from sklearn import cross_validation

from hmmlearn import hmm


####define filename#################################################################################################################################################


numstring = "1"

##################

confufilename = "Confusion_Matrix" + str(numstring)


traindatafilename = 'train'+numstring+'.csv'
trainlabeldatafilename = 'label_train'+numstring+'.csv'



#####support function############################################################################################


def Confusion_matrix_plot(clf_name, prediction, y_test, classes = ['1', '2', '3','4','R','W'], cmap=plt.get_cmap('Blues')):
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








# loading data #########################################################
# training data
dataframe = datanize((traindatafilename),header=None,firstnoneed=0)

X = dataframe.values


X = (X / 10000).astype(np.float32)


labelframe = datanize((trainlabeldatafilename),header=None,firstnoneed=0)

Y = labelframe.values

Y = Y.T[0]

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# Training #######################################################################################################################################33


#LR #######################################################################

logistic = linear_model.LogisticRegression()
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)
print("Logistic regression using raw pixel features:\n%s\n" % (
    classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))

Confusion_matrix_plot('Logistic regression', logistic_classifier.predict(X_test), Y_test, classes = ['1', '2', '3','4','R','W'], cmap=plt.get_cmap('Blues'))

resultLR = logistic_classifier.predict(X_test)

#print(result)


#DBN ####################################################################3


classifier = SupervisedDBNClassification(hidden_layers_structure=[500, 500],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
_ = classifier.fit(X_train, Y_train)


print(classification_report(Y_test, classifier.predict(X_test)))

Confusion_matrix_plot('DBN', classifier.predict(X_test), Y_test, classes = ['1', '2', '3','4','R','W'], cmap=plt.get_cmap('Blues'))


resultDBN = classifier.predict(X_test)





####combination######################################

listLR = [0,2,3]
listDBN = [1,4,5]
final = []

for i in resultLR:
    if i not in listLR:
        final.append("TODO")
    if i in listLR:
        final.append(i)

index = 0

for i in final:
    if i == "TODO":
        final[index] = resultDBN[index]
    index += 1


print(classification_report(Y_test, final))

Confusion_matrix_plot('Combination', final, Y_test, classes = ['1', '2', '3','4','R','W'], cmap=plt.get_cmap('Blues'))

###HMM model######################################


#print(final)

final = np.transpose(np.array(final)[np.newaxis])




model = hmm.GaussianHMM(n_components=6, covariance_type="full", n_iter=1000)
model.fit(final)

hidden_states = model.predict(final)



############################3
def errorestimate(num, obnum, hidden_states, observation):
    count = 0
    sameones = 0
    for i,part in enumerate(hidden_states):
        if part == num:
            count += 1
            if observation[i] == obnum:
                sameones += 1
    prob = sameones/(count+1)
    return prob

def findcorrect(hidden_states, observation):
    label = [0,1,2,3,4,5]
    dic ={}
    for i in label:
        probvalue = []
        for j in label:
            prob = errorestimate(i, j, hidden_states, observation)
            probvalue.append(prob)
        newindex = probvalue.index(max(probvalue))
        dic[i] = newindex

    hidden_states_new = []
    for part in hidden_states:
        hidden_states_new.append(dic[part])

    return hidden_states_new


hidden_states = findcorrect(hidden_states, final)



##prediction #########################

print(classification_report(Y_test, hidden_states))

Confusion_matrix_plot('Combination_HMM', hidden_states, Y_test, classes = ['1', '2', '3','4','R','W'], cmap=plt.get_cmap('Blues'))


