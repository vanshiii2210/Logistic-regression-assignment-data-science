import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

# Set page configuration
st.set_page_config(page_title="Titanic Data Analysis", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Data Analysis and Prediction")
st.markdown("---")

# # 1. Data Exploration:
st.header("1. Data Exploration")


train_data=pd.read_csv('Titanic_train.csv')

st.subheader("Training Data Overview")
st.dataframe(train_data)

st.subheader("Data Info")
st.text(f"Dataset shape: {train_data.shape}")

st.subheader("Data Description")
st.dataframe(train_data.describe())


st.subheader("Data Distribution - Histograms")
fig1, ax1 = plt.subplots(figsize=(12, 8))
train_data.hist(bins=20, ax=ax1, figsize=(12, 8))
plt.tight_layout()
st.pyplot(fig1)


st.subheader("Boxplots for Outlier Detection")
fig2, ax2 = plt.subplots(figsize=(15, 8))
train_data.boxplot(ax=ax2, figsize=(15, 8))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)


st.subheader("Pairplot Analysis")
fig3 = sns.pairplot(train_data)
st.pyplot(fig3)


st.subheader("Gender Distribution")
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.countplot(x='Sex', data=train_data, palette='plasma', ax=ax4)
ax4.set_title('Gender Distribution in Training Data')
plt.tight_layout()
st.pyplot(fig4)

st.markdown("---")

# # 2. Data Preprocessing:
st.header("2. Data Preprocessing")


train_data.drop('Cabin',inplace=True,axis=1)

st.subheader("Data after removing Cabin column")
st.dataframe(train_data)

st.subheader("Missing Values Count")
missing_values = train_data.isnull().sum()
st.dataframe(missing_values[missing_values > 0].to_frame().T)


#treating null values
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode().iloc[0],inplace=True)

st.subheader("Missing Values After Treatment")
missing_after = train_data.isnull().sum()
st.write("Missing values count:", missing_after.sum())


train_data.drop(columns=['Name','Ticket'],inplace=True)

train_data=pd.get_dummies(train_data,columns=['Sex','Embarked'],dtype=int)

st.subheader("Final Processed Training Data")
st.dataframe(train_data)

st.markdown("---")

# ### Test Data
st.header("Test Data Processing")


test_data=pd.read_csv('Titanic_test.csv')

st.subheader("Test Data Overview")
st.dataframe(test_data)

test_data.drop('Cabin',inplace=True,axis=1)

st.subheader("Missing Values in Test Data")
test_missing = test_data.isnull().sum()
st.dataframe(test_missing[test_missing > 0].to_frame().T)


test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

test_data['Embarked'].fillna(test_data['Embarked'].mode().iloc[0],inplace=True)

test_data.drop(columns=['Name','Ticket'],inplace=True)

test_data=pd.get_dummies(test_data,columns=['Sex','Embarked'],dtype=int)

st.subheader("Processed Test Data")
st.dataframe(test_data)

st.subheader("Test Data Distribution")
fig5, ax5 = plt.subplots(figsize=(12, 8))
test_data.hist(bins=20, ax=ax5, figsize=(12, 8))
plt.tight_layout()
st.pyplot(fig5)

st.markdown("---")

# # 3. Model Building:
st.header("3. Model Building")


features_train = train_data.drop('Survived',axis=1)
target_train = train_data[['Survived']]

st.subheader("Features and Target")
st.write("Features shape:", features_train.shape)
st.write("Target shape:", target_train.shape)





from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test=train_test_split(features_train,target_train,train_size=0.8,random_state=11)

logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)

st.subheader("Model Training Complete")
st.write("Training set size:", x_train.shape)
st.write("Test set size:", x_test.shape)

st.markdown("---")

# # 4. Model Evaluation:
st.header("4. Model Evaluation")


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,roc_auc_score,roc_curve

cm = confusion_matrix(y_test,y_pred)
st.subheader("Confusion Matrix")
st.dataframe(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))

st.subheader("Classification Report")
report = classification_report(y_test,y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

accuracy = accuracy_score(y_test,y_pred)*100
st.subheader("Model Accuracy")
st.metric("Accuracy", f"{accuracy:.2f}%")


roc_auc = roc_auc_score(y_test,y_pred)
st.metric("ROC AUC Score", f"{roc_auc:.4f}")

fpr,tpr,thr=roc_curve(y_test,y_pred)

st.subheader("ROC Curve")
fig6, ax6 = plt.subplots(figsize=(8, 6))
ax6.plot(fpr,tpr,label='Logistic Regression Model', linewidth=2)
ax6.set_xlabel('False Positive Rate')
ax6.set_ylabel('True Positive Rate')
ax6.set_title('ROC Curve')
ax6.plot([0,1],[0,1],color='red',linestyle='--', label='Random Classifier')
ax6.legend()
ax6.grid(True, alpha=0.3)
st.pyplot(fig6)

st.markdown("---")

# # 5. Interpretation:
st.header("5. Model Interpretation")


st.subheader("Model Parameters")
st.write("Intercept:", logreg.intercept_[0])
st.write("Number of coefficients:", len(logreg.coef_[0]))

# Show feature importance
coef_df = pd.DataFrame({
    'Feature': features_train.columns,
    'Coefficient': logreg.coef_[0]
})
coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

st.subheader("Feature Importance")
fig7, ax7 = plt.subplots(figsize=(10, 8))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', ax=ax7)
ax7.set_title('Feature Importance (Logistic Regression Coefficients)')
plt.tight_layout()
st.pyplot(fig7)

st.markdown("---")

#prediction on titanic test data
st.header("6. Predictions on Test Data")


st.subheader("Final Data Check")
final_missing = test_data.isnull().sum()
if final_missing.sum() > 0:
    st.write("Remaining missing values:")
    st.dataframe(final_missing[final_missing > 0].to_frame().T)
    test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
    st.write("Filled missing Fare values with mean")
else:
    st.write("No missing values in test data ✅")

prediction=logreg.predict(test_data)

predictions_df = pd.DataFrame(prediction, index=test_data.index, columns=['Survival_Prediction'])
result = test_data.join(predictions_df)

st.subheader("Predictions Results")
st.dataframe(result)

# Summary statistics
survival_count = pd.Series(prediction).value_counts()
st.subheader("Prediction Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Deaths", survival_count.get(0, 0))
with col2:
    st.metric("Predicted Survivors", survival_count.get(1, 0))

st.markdown("---")

# # Interview Questions:
# st.header("7. Key Concepts & Interview Questions")


# with st.expander("📚 Key Machine Learning Concepts"):
#     st.markdown("""
#     **1. What is the difference between precision and recall?**
#     - **Precision**: Measures the accuracy of positive predictions. High precision means few false positives.
#     - **Recall**: Measures how well the model identifies all actual positives. High recall means few false negatives.
    
#     **2. What is cross-validation, and why is it important in binary classification?**
#     - A technique to evaluate model performance by splitting data into multiple subsets (folds) and testing the model across these folds.
#     - Important for reducing overfitting and ensuring the model generalizes well to new data.
    
#     **3. What does the ROC curve tell us?**
#     - Shows the trade-off between True Positive Rate and False Positive Rate
#     - AUC (Area Under Curve) closer to 1.0 indicates better model performance
#     - Helps in selecting optimal threshold for classification
#     """)
