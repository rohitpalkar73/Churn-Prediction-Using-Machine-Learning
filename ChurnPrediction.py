# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#%matplotlib inline
import pickle

# Load Dataset
inputPath = "Churn_Modelling.csv"
dataset = pd.read_csv(inputPath, header=0)
dataset.head()

# Exploratory Analysis
dataset.describe()

# Categorical data points exploration
# Gender, Geography are the useful data points, where as surname is of no significance for the model.

dataset.groupby("Gender")["Geography"].count()
dataset.groupby("Geography")["Gender"].count()
# Conversion of categorical values into numerical levels
dataset["Gender1"] = dataset["Gender"]
dataset["Gender"] = pd.Categorical(dataset["Gender"])
dataset["Gender"] = dataset["Gender"].cat.codes
dataset.head()

dataset["Geography1"] = dataset["Geography"]
dataset["Geography"] = pd.Categorical(dataset["Geography"])
dataset["Geography"] = dataset["Geography"].cat.codes
dataset.head()

# Age binning
dataset["AgeBin"] = pd.cut(dataset['Age'], [0, 16, 32,48,64,500])
dataset["AgeBin"] = pd.Categorical(dataset["AgeBin"])
dataset["AgeBin"] = dataset["AgeBin"].cat.codes
dataset.loc[dataset["Age"] > 60].head()

# Binning credit score
dataset['CreditScoreBin'] = pd.cut(dataset['CreditScore'], [0, 450, 550,650,750,900])

dataset["CreditScoreBin"] = pd.Categorical(dataset["CreditScoreBin"])
dataset["CreditScoreBin"] = dataset["CreditScoreBin"].cat.codes
dataset.head()

# Binning Balance
dataset['BalanceBin'] = pd.cut(dataset['Balance'], [-1, 50000, 100000,150000,200000,1000000000000000])

dataset["BalanceBin"] = pd.Categorical(dataset["BalanceBin"])
dataset["BalanceBin"] = dataset["BalanceBin"].cat.codes
dataset.head()

# Binning Estimated Salary
dataset['EstimatedSalaryBin'] = pd.cut(dataset['EstimatedSalary'], [-1, 50000, 100000,150000,200000,1000000000000000])

dataset["EstimatedSalaryBin"] = pd.Categorical(dataset["EstimatedSalaryBin"])
dataset["EstimatedSalaryBin"] = dataset["EstimatedSalaryBin"].cat.codes
dataset.head()

# Box plot
fig, ((a,b,c,d),(e,f,g,h)) = plt.subplots(2,4)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=1, hspace=0.6)

a.set_title("Age")
a.boxplot(dataset["Age"])
b.set_title("CreditScore")
b.boxplot(dataset["CreditScore"])
c.set_title("Tenure")
c.boxplot(dataset["Tenure"])
d.set_title("Balance")
d.boxplot(dataset["Balance"])
e.set_title("NumOfProducts")
e.boxplot(dataset["NumOfProducts"])
f.set_title("HasCrCard")
f.boxplot(dataset["HasCrCard"])
g.set_title("IsActiveMember")
g.boxplot(dataset["IsActiveMember"])
h.set_title("EstimatedSalary")
h.boxplot(dataset["EstimatedSalary"])
plt.show()

# Correlation
dataset.corr()["Exited"]

# Remove the non-necessary fields
dataset1 = dataset.copy()
dataset = dataset.drop(["Geography"], axis=1)
dataset = dataset.drop(["CustomerId"], axis=1)
dataset = dataset.drop(["Gender1"], axis=1)
dataset = dataset.drop(["Geography1"], axis=1)
dataset = dataset.drop(["Age"], axis=1)
dataset = dataset.drop(["CreditScore"], axis=1)
dataset = dataset.drop(["Balance"], axis=1)
dataset = dataset.drop(["EstimatedSalary"], axis=1)
dataset = dataset.drop(["Surname"], axis=1)
dataset = dataset.drop(["RowNumber"], axis=1)

# Random shuffle of records 
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Split data into train and test datasets
test_data_split = 0.2
x_train,x_test , y_train, y_test = train_test_split(dataset.drop(["Exited"],axis=1),dataset["Exited"],test_size = test_data_split)
x_test.describe()

# feature importance analysis
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x_train, y_train)

# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
d = {'columnName': x_train.columns.values, 'featureScore': fit.scores_}
df = pd.DataFrame(data=d)
df.sort_values(['featureScore'], ascending=False)

# Check the distribution of Exited in train & Test datasets
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=1, hspace=1)
plt.subplot(1,3,1)
y_train.iloc[:].value_counts().plot(kind = 'bar',title="train Dataset")
plt.subplot(1,3,2)
y_test.iloc[:].value_counts().plot(kind = 'bar',title="test Dataset")
plt.subplot(1,3,3)
dataset.iloc[:,7].value_counts().plot(kind = 'bar',title="Complete Dataset")
plt.show()

# Build model

# Decision Trees

from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
clf = tree.DecisionTreeClassifier(random_state=0, max_depth = 10, max_leaf_nodes=None)
clf.fit(x_train, y_train)
print(clf)

# Predict test data
y_pred = clf.predict(x_test)

# Predict
y_pred_dt = clf.predict(x_test)

# Accuracy metrics
acc_log = accuracy_score(y_test,y_pred)
print("accuracy:", acc_log)
# Confusion Matrix
confusion_matrix(y_test, y_pred)

# ROC Curve for Decision Tree

fpr = dict()
tpr = dict()
roc_auc = dict()

y_pred_proba = clf.predict_proba(x_test)[::,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_auc = auc(fpr,tpr)
print(roc_auc)

plt.title("ROC Curve (Decision Tree)")
plt.plot(fpr, tpr, color='red',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.show()

#dataset.columns

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))