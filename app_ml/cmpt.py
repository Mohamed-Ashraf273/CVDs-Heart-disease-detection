import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import  models
from tensorflow.keras.layers import  Dense, Dropout


# Read data
data = pd.read_csv('student_version.csv')
train_Data = data.iloc[:, :-1]
target_data = data.iloc[:, -1]
KNN_classes = np.unique(target_data)


# Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(train_Data, target_data, test_size=0.1, random_state=42)
test_data = pd.read_csv('X_test.csv')
id_column = test_data.iloc[:, 0]
x_test_cases = test_data.iloc[:, 1:-1]


# # One-hot encoding categorical data using Pandas
x_train_encoded = pd.get_dummies(x_train)
x_test_cases_encoded = pd.get_dummies(x_test_cases)
x_test_cases_encoded = x_test_cases_encoded.reindex(columns=x_train_encoded.columns, fill_value=0)
x_test_encoded = pd.get_dummies(x_test)
# Create pair plot
sns.pairplot(data, hue='HeartDisease', palette='viridis', markers=["o", "s"], plot_kws={'alpha':0.6})

# Show plot
plt.show()

# # Scaling
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(x_train_encoded)
x_test_cases_encoded = scalar.transform(x_test_cases_encoded)
X_test_scaled = scalar.transform(x_test_encoded)

# # Defining Accuracies of models
KNN_accuracy = 0.0
LG_accuracy = 0.0
NN_accuracy = 0.0
KNN_call = 0.0
LG_call = 0.0
NN_call = 0.0



# ######################################################################################################################################
# Get best n_neighbors
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]
}
model = KNeighborsClassifier(p=2, metric='minkowski')
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_n_neighbors = grid_search.best_params_['n_neighbors']

# Train model (KNN)
KNN_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, p=2, metric='minkowski')
KNN_model.fit(X_train_scaled, y_train)


y_KNN_predict = KNN_model.predict(X_test_scaled)
KNN_accuracy = accuracy_score(y_test, y_KNN_predict)
KNN_call = recall_score(y_test, y_KNN_predict, average='binary')
conf_matrix = confusion_matrix(y_test, y_KNN_predict)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds',
            xticklabels=KNN_classes,  
            yticklabels=KNN_classes)  
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('KNN_confusion_matrix.png')  
plt.close()



# Evalusting KNN_model
y_KNN_predict1 = KNN_model.predict(x_test_cases_encoded)
# Add predictions to a DataFrame with the IDs
results_df_KNN = pd.DataFrame({
    'ID': id_column,
    'HeartDisease': y_KNN_predict1
})

results_df_KNN.to_csv('KNN_predictions.csv', index=False)
# ######################################################################################################################################
# Train a Logistic regression model (ML Approach)
LG_model = LogisticRegression()
LG_model.fit(X_train_scaled, y_train)
y_LG_predict = LG_model.predict(X_test_scaled)
LG_accuracy = accuracy_score(y_test, y_LG_predict)
LG_call = recall_score(y_test, y_LG_predict, average='binary')
conf_matrix_log_reg = confusion_matrix(y_test, y_LG_predict)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('LG_confusion_matrix.png')  
plt.close()


y_LG_predict1 = LG_model.predict(x_test_cases_encoded)
# Add predictions to a DataFrame with the IDs
results_df_LG = pd.DataFrame({
    'ID': id_column,
    'HeartDisease': y_LG_predict1
})

results_df_LG.to_csv('LG_predictions.csv', index=False)
# ######################################################################################################################################
# Train NN model (DL Approach)
NN_model = models.Sequential()
NN_model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
NN_model.add(Dense(64, activation='relu'))# another hidden layer
NN_model.add(Dense(1, activation='sigmoid'))# sigmoid because it is binary not multiclass, if it was multi then softmax
NN_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
no_epochs = 15
batch_size = 15

history = NN_model.fit(
    X_train_scaled,                   
    y_train,           
    batch_size=batch_size,     
    epochs=no_epochs,          
    validation_split=0.1          
)  
y_NN_predict = NN_model.predict(X_test_scaled)
y_NN_predict = (y_NN_predict > 0.5).astype(int) 
NN_accuracy = accuracy_score(y_test, y_NN_predict)
NN_call = recall_score(y_test, y_NN_predict, average='binary')
confusion_mtx = tf.math.confusion_matrix(y_test, y_NN_predict)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Reds', 
            xticklabels=['0', '1'], 
            yticklabels=['0', '1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('NN_confusion_matrix.png')  
plt.close()



y_NN_predict1 = LG_model.predict(x_test_cases_encoded)
# Add predictions to a DataFrame with the IDs
results_df_NN = pd.DataFrame({
    'ID': id_column,
    'HeartDisease': y_NN_predict1
})

results_df_NN.to_csv('NN_predictions.csv', index=False)
# ######################################################################################################################################
# # For KNN prediction
# def predict_KNN(input_data):
#     pp_data = pd.get_dummies(input_data)
#     pp_data = pp_data.reindex(columns=x_train_encoded.columns, fill_value=0)
#     pp_data_scaled = scalar.transform(pp_data)
#     y_predict = KNN_model.predict(pp_data_scaled)
#     return y_predict

# # For LG prediction
# def predict_ML(input_data):
#     pp_data = pd.get_dummies(input_data)
#     pp_data = pp_data.reindex(columns=x_train_encoded.columns, fill_value=0)
#     pp_data_scaled = scalar.transform(pp_data)
#     y_predict = LG_model.predict(pp_data_scaled)
#     return y_predict

# # For NN prediction
# def predict_DL(input_data):
#     pp_data = pd.get_dummies(input_data)
#     pp_data = pp_data.reindex(columns=x_train_encoded.columns, fill_value=0)
#     pp_data_scaled = scalar.transform(pp_data)
#     y_predict = NN_model.predict(pp_data_scaled)
#     return y_predict

# # Get KNN accuracy
# def Get_KNN_Accuracy():
#     return KNN_accuracy

# # Get LG accuracy
# def GET_LG_Accuracy():
#     return LG_accuracy

# # Get NN accuracy
# def Get_NN_Accuracy():
#     return NN_accuracy


with open('model_accuracies.txt', 'w') as f:
    f.write(f'KNN Model Accuracy: {KNN_accuracy:.4f}\n')
    f.write(f'LG Model Accuracy: {LG_accuracy:.4f}\n')
    f.write(f'NN Model Accuracy: {NN_accuracy:.4f}\n')


with open('model_recall.txt', 'w') as f:
    f.write(f'KNN Model Recall: {KNN_call:.4f}\n')
    f.write(f'LG Model Recall: {LG_call:.4f}\n')
    f.write(f'NN Model Recall: {NN_call:.4f}\n')


