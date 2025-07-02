#!/usr/bin/env python
# coding: utf-8

# # Synthetic Data Analysis and Classification Model

# ## 1. Load Data and Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('synthetic_data.csv')

df.head()


# ## 2. Exploratory Data Analysis (EDA)

# In[ ]:


df.info()


# In[ ]:


df.describe()


# ### Target Variable Distribution

# In[ ]:


sns.countplot(x='Purchase', data=df)
plt.title('Distribution of Purchase Variable')
plt.show()


# ### Feature Distributions

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(df['Age'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution')

sns.histplot(df['Income'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Income Distribution')

sns.countplot(x='Education', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Education Level Distribution')

sns.histplot(df['HoursWorked'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Hours Worked Distribution')

plt.tight_layout()
plt.show()


# ### Correlation Matrix

# In[ ]:


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# ## 3. Feature Engineering (Minimal for this example)

# In[ ]:


# No explicit feature engineering for this simple example.
# Education is already numerical. If it were categorical strings, we'd encode it.


# ## 4. Model Training

# In[ ]:


X = df.drop('Purchase', axis=1)
y = df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[ ]:


print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")


# In[ ]:


# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)


# ## 5. Model Evaluation

# In[ ]:


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# In[ ]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:


print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ## 6. Feature Importance

# In[ ]:


importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.show()

print(feature_importance_df)


# End of Analysis.
