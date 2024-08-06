import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import mlflow
import mlflow.sklearn
import optuna
from optuna.visualization import plot_optimization_history

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
df = pd.read_csv("./data/cardio_train_full.csv", delimiter=';')
df.info()
print(df.head(5))
print(df.isna().sum())

#drop id column
df = df.drop('id', axis=1)

# Visualization
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), linewidth=.02, annot=True, cmap="coolwarm")
plt.savefig('HD correlation.png')

df.hist(figsize=(12,12))
plt.savefig('featuresplot.png')

# Split data
x = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)

# Define Optuna objective function
def objective(trial):
    #making objective function nested to avoid error
    #for overwriting logs with same flowid
    mlflow.start_run(nested=True)
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion='entropy',
        max_depth=max_depth,
        )
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics and model to MLflow
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_params(trial.params)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(classifier, "model")
    mlflow.end_run()  # End the nested run
    return accuracy

# Optuna optimization within an MLflow run
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("hd_model_experiment")

with mlflow.start_run(run_name='hd_model_exp_02'):
    print('+++MlFlow Started ++++++++')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print('+++MlFlow Completed ++++++++')
    mlflow.end_run()
print('Results ++++++++++++++++++++++++++++++')
# Best trial results
print("Best trial:", study.best_trial.params)
print("Best accuracy:", study.best_trial.value)

# Save the best model
best_classifier = RandomForestClassifier(**study.best_trial.params)
best_classifier.fit(X_train, y_train)
with open('./app/random_forest_model.pkl', 'wb') as file:
    pickle.dump(best_classifier, file)

# Final model evaluation
y_pred = best_classifier.predict(X_test)
print(classification_report(y_test, y_pred))