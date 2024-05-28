#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings("ignore", message=".*the default behavior of `mode`.*")


# In[31]:


data = pd.read_csv("dataset.csv")
print(data.head())
print(data.describe())
print("info")
print(data.info())
print(data.isnull().sum())
print(data.duplicated().sum())


# In[32]:



missing_values = data.isnull().sum()
total_missing = missing_values.sum()
missing_percent = (missing_values / data.shape[0]) * 100
print("Missing values:\n", missing_values)
print("Missing values percentage:\n", missing_percent)


plt.figure(figsize=(10, 6))
missing_percent.plot(kind='bar', color='red')
plt.title('Proportion of Missing Values by Feature')
plt.xlabel('Features')
plt.ylabel('Percentage of Missing Values')
plt.show()

print("Duplicates:", data.duplicated().sum())


# In[33]:



data.fillna(data.mean(), inplace=True)


data.drop_duplicates(inplace=True)


# In[34]:


from scipy.stats import zscore


numeric_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                   'DiabetesPedigreeFunction', 'Age', 'HbA1c Levels', 'Stress Levels',
                   'Sleep Quality', 'Family History']
z_scores = np.abs(zscore(data[numeric_columns]))
print("Z-scores:\n", z_scores)
threshold = 3
outliers = np.where(z_scores > threshold)
data_cleaned = data[(z_scores < threshold).all(axis=1)]

print("Original data shape:", data.shape)
print("Cleaned data shape:", data_cleaned.shape)


#scaler = StandardScaler()
#data_cleaned[numeric_columns] = scaler.fit_transform(data_cleaned[numeric_columns])

#print("Normalized data:\n", data_cleaned)


# In[35]:



correlations = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, cmap='coolwarm', annot=True)
plt.title('Correlation Matrix')
plt.show()


plt.figure(figsize=(10, 6))


# In[36]:


# Create scatter plot with different colors for each outcome
scatter = plt.scatter(data['Glucose'], data['BMI'], c=data['Outcome'], cmap='coolwarm', label=data['Outcome'])
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('Scatter plot of Glucose vs BMI')

# Create a custom legend
legend1 = plt.legend(*scatter.legend_elements(), title='Outcome')
plt.gca().add_artist(legend1)
plt.show()


# In[37]:


np.random.seed(42)

kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Glucose', 'BMI']])
centers = kmeans.cluster_centers_

# Function to draw an ellipse around clusters
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 1.2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 1.2 * np.sqrt(covariance)
    
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle,facecolor='none', **kwargs))

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Before KMeans
ax1.scatter(data['Glucose'], data['BMI'], c=data['Outcome'],cmap='coolwarm', s=30,label=data['Outcome'])
ax1.set_title('Before KMeans')
ax1.set_xlabel('Glucose')
ax1.set_ylabel('BMI')


# After KMeans
colors = ['red','yellow']
for i in range(2):
    points = data[data['Cluster'] == i]
    ax2.scatter(points['Glucose'], points['BMI'], s=30, color=colors[i], label=f'Cluster {i}')
    draw_ellipse(centers[i], np.cov(points[['Glucose', 'BMI']].values.T), ax=ax2, edgecolor=colors[i], alpha=0.5)

ax2.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.6, marker='X', label='Centroids')
ax2.set_title('After KMeans')
ax2.set_xlabel('Glucose')
ax2.set_ylabel('BMI')
ax2.legend()
plt.show()


# In[38]:


X = data.drop(['Outcome', 'Pregnancies',], axis=1)
y = data['Outcome']

from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information
mutual_info = mutual_info_classif(X, y)
mutual_info_series = pd.Series(mutual_info, index=X.columns)

# Sort and display mutual information scores
mutual_info_series = mutual_info_series.sort_values(ascending=False)
print("Mutual information scores:\n", mutual_info_series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[39]:


log_reg_model = LogisticRegression(max_iter=1000)  # Increase max_iter
log_reg_model.fit(X_train, y_train)
log_reg_accuracy = log_reg_model.score(X_test, y_test)
log_reg_pred = log_reg_model.predict(X_test)
log_reg_cm = metrics.confusion_matrix(y_test, log_reg_pred)
print("Logistic regression accuracy",log_reg_accuracy)
print("confusion matrix of logistic regression",log_reg_cm)
print("prediction of logistic regression",log_reg_pred)
print("Intercept: ", log_reg_model.intercept_)
print("Coefficients: ", log_reg_model.coef_)
plt.figure(figsize=(2, 2))
sns.heatmap(log_reg_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()


# In[40]:


# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_accuracy = dt_model.score(X_test, y_test)
dt_pred = dt_model.predict(X_test)
dt_cm = metrics.confusion_matrix(y_test, dt_pred)
print("confusion matrix of Decision Tree",dt_cm)
print("Prediction of Decision Tree",dt_pred)
print("accuracy of decision tree",dt_accuracy)
plt.figure(figsize=(2, 2))
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix decision Tree")
plt.show()


# In[41]:



import warnings
warnings.filterwarnings("ignore", message=".*the default behavior of `mode`.*")
knn_model = KNeighborsClassifier()

knn_model.fit(X_train, y_train)
knn_accuracy = knn_model.score(X_test, y_test)
knn_pred = knn_model.predict(X_test)
knn_cm = metrics.confusion_matrix(y_test, knn_pred)
print("confusion matrix of KNeighour Classification")
print(knn_cm)
plt.figure(figsize=(2, 2))
print("accuracy of KNeighour Classification",knn_accuracy)
print("KNeighour Classification prediction",knn_pred)
sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix - K-Nearest Neighbors Classifier")
plt.show()


# In[42]:


models_accuracy = {'Logistic Regression': log_reg_accuracy,
                   'Decision Tree': dt_accuracy,
                   'KNN': knn_accuracy}
print(models_accuracy)


# In[43]:


import warnings
warnings.filterwarnings("ignore", message=".*the default behavior of `mode`.*")

# Assuming the models and accuracies are already defined somewhere earlier in the code
X_test = X_test[X_train.columns]
best_model_name = max(models_accuracy, key=models_accuracy.get)
print(f"Best Model: {best_model_name}")

# Select the best model
if best_model_name == 'Logistic Regression':
    best_model = log_reg_model
elif best_model_name == 'Decision Tree':
    best_model = dt_model
else:
    best_model = knn_model

# Print confusion matrix
""""print(f"Confusion Matrix for {best_model_name}:")
if best_model_name == 'Logistic Regression':
    print(log_reg_cm)
elif best_model_name == 'Decision Tree':
    print(dt_cm)
else:
    print(knn_cm)"""



def risk_level(glucose):
    if glucose > 140:
        return "High"
    elif 100 < glucose <= 140:
        return "Medium"
    else:
        return "Low"
    data['Risk_Level'] = data['Glucose'].apply(risk_level)

# Define function to predict diabetes
def predict_diabetes(model):
    Pregnancies = float(input("Enter Pregnancies: "))
    glucose = float(input("Enter Glucose level: "))
    blood_pressure = float(input("Enter Blood Pressure: "))
    skin_thickness = float(input("Enter Skin Thickness: "))
    insulin = float(input("Enter Insulin: "))
    bmi = float(input("Enter BMI: "))
    diabetes_pedigree = float(input("Enter Diabetes Pedigree Function: "))
    age = float(input("Enter Age: "))
    hba1c_levels = float(input("Enter HbA1c Levels: "))
    stress_levels = float(input("Enter Stress Levels: "))
    sleep_quality = float(input("Enter Sleep Quality: "))
    family_history = float(input("Enter Family History: "))
  
    new_data = [[Pregnancies,glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age, hba1c_levels, stress_levels, sleep_quality, family_history]]
  
    prediction = model.predict(new_data)
    risk = risk_level(glucose)

    if prediction[0] == 1:
        print("Prediction: Person has diabetes")
        print("Risk Level:", risk)
    else:
        print("Prediction: Person does not have diabetes")

import warnings
warnings.filterwarnings("ignore", message=".*the default behavior of `mode`.*")


predict_diabetes(best_model)

X_test['Predicted_Diabetes'] = best_model.predict(X_test)
predicted_data = pd.concat([X_test, y_test], axis=1)
predicted_data.to_csv("dm_project_final_predicted.csv", index=False)

