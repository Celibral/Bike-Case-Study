import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt

def eval(y_test, preds):
    return(accuracy_score(y_test, preds), (1 - (accuracy_score(y_test, preds))), roc_auc_score(y_test, preds),   precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds))


df = pd.read_csv('df_data_cleaned.csv', index_col=[0])

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Purchased Bike', axis=1), # predictive variables
    df['Purchased Bike'], # target
    test_size=0.3, # portion of dataset to allocate to test set
    random_state=0, # we are setting the seed here
)

scaler = MinMaxScaler()

#  fit  the scaler to the train set
scaler.fit(X_train) 

X_train = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_train.columns
)

# feature selection
from sklearn.ensemble import RandomForestClassifier

feature_names = [X_train.columns]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


X_train = X_train[['Income', 'Children', 'Cars', 'Commute Distance', 'Age']]
X_test = X_test[['Income', 'Children', 'Cars', 'Commute Distance', 'Age']]

# XG Boost

# XG Boost
'''
param_grid = {
    "max_depth": [3, 4, 5, 7, 8, 9 , 10],
    "learning_rate": [0.1, 0.01, 0.05],
    "gamma": [0, 0.25, 0.5, 1],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.1, 0.35, 0.5, 0.8],
    "colsample_bytree": [0.25, 0.5, 0.75],
}

# Init classifier
xgb_cl = xgb.XGBClassifier(objective="binary:logistic")

# Init Grid Search
grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")

# Fit
_ = grid_cv.fit(X_train, y_train)

grid_cv.best_score_

grid_cv.best_params_
'''

model = xgb.XGBClassifier(objective='binary:logistic',
                          booster='gbtree',
                          eval_metric='auc',
                          tree_method='hist',
                          grow_policy='lossguide',
                          use_label_encoder=False)

_ = model.fit(X_train, y_train)

preds = model.predict(X_test)
y_proba = model.predict_proba(X_test)

#define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)

#use k-fold CV to evaluate model
scores = cross_val_score(model, X_test, y_test, scoring='accuracy',
                         cv=cv, n_jobs=-1)


print(scores.mean())

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,2):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
lw = 0
plt.plot(
    fpr[0],
    tpr[0],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[0],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

# calibration plot
skplt.metrics.plot_calibration_curve(y_test, [y_proba])
