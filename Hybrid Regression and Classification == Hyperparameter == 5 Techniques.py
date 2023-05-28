import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load ADNI training dataset as a Pandas dataframe
adni_train_df = pd.read_csv('AD_train.csv')

# Load ADNI testing dataset as a Pandas dataframe
adni_test_df = pd.read_csv('AD_test.csv')

# Encode the Status variable in training dataset
le = LabelEncoder()
adni_train_df['Status'] = le.fit_transform(adni_train_df['Status'])

# Encode the Status variable in testing dataset
adni_test_df['Status'] = le.transform(adni_test_df['Status'])

# Split training data into X and y
X_train = adni_train_df.iloc[:, 1:]
y_train = adni_train_df['Status']

# Split testing data into X and y
X_test = adni_test_df.iloc[:, 1:]
y_test = adni_test_df['Status']

# Define hyperparameters for the models
model_hyperparameters = {
    'rf': {'n_estimators': 100},
    'svm': {'kernel': 'rbf', 'C': 1, 'probability': True},
    'knn': {'n_neighbors': 5},
    'lr': {'solver': 'liblinear', 'C': 1.0},
    'gb': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
}

# Train with Random Forest
rf_model = RandomForestClassifier(**model_hyperparameters['rf'])
rf_model.fit(X_train, y_train)

#print("Random Forest Feature Importances:")
#print(dict(zip(X_train.columns, rf_model.feature_importances_)))

# Train with SVM
svm_model = SVC(**model_hyperparameters['svm'])
svm_model.fit(X_train, y_train)

# Train with K-Nearest Neighbors
knn_model = KNeighborsClassifier(**model_hyperparameters['knn'])
knn_model.fit(X_train, y_train)

# Train with Logistic Regression
lr_model = LogisticRegression(**model_hyperparameters['lr'])
lr_model.fit(X_train, y_train)

# Train with Gradient Boosting
gb_model = GradientBoostingClassifier(**model_hyperparameters['gb'])
gb_model.fit(X_train, y_train)

# Combine predictions from all models
rf_pred = rf_model.predict_proba(X_test)
svm_pred = svm_model.predict_proba(X_test)
knn_pred = knn_model.predict_proba(X_test)
lr_pred = lr_model.predict_proba(X_test)
gb_pred = gb_model.predict_proba(X_test)

# Combine probabilities from all models
hybrid_pred = (rf_pred + svm_pred + knn_pred + lr_pred + gb_pred) / 5

# Evaluate hybrid model with testing set
print("Hybrid Classification Metrics:")
print("Accuracy:", accuracy_score(y_test, hybrid_pred.argmax(axis=1)))
print("Precision:", precision_score(y_test, hybrid_pred.argmax(axis=1), average='weighted'))
print("Recall:", recall_score(y_test, hybrid_pred.argmax(axis=1), average='weighted'))
print("F1 Score:", f1_score(y_test, hybrid_pred.argmax(axis=1), average='weighted'))


# Compute and plot ROC curve for hybrid model
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(le.classes_)):
    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), hybrid_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(pd.get_dummies(y_test).values.ravel(), hybrid_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=lw, label='Micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(len(le.classes_)):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(le.inverse_transform([i])[0], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Hybrid Model')
plt.legend(loc="lower right")
plt.show()
############################################################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load training data
train_dataset=pd.read_csv("AD_train.csv")

# Label encoding
le = LabelEncoder() 
train_dataset['Status'] = le.fit_transform(train_dataset['Status']) 
X_train = train_dataset.drop(['Status'] , axis=1)
y_train = train_dataset['Status']

# Scale features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Load testing data
test_dataset=pd.read_csv("AD_test.csv")

# Label encoding
test_dataset['Status'] = le.transform(test_dataset['Status']) 
X_test = test_dataset.drop(['Status'] , axis=1)
y_test = test_dataset['Status']

# Scale features using StandardScaler
X_test = scaler.transform(X_test)

# Train with Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Train with SVM
svm_model = SVR(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# Train with K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train with Decision Tree
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)

# Train with Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Combine predictions from all models
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Combine predictions from all models
hybrid_pred = (rf_pred + svm_pred + knn_pred + dt_pred + gb_pred) / 5

# Evaluate hybrid model with testing set
print("Hybrid Regression Metrics:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, hybrid_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, hybrid_pred))
print("R-squared (R^2) Score:", r2_score(y_test, hybrid_pred))
