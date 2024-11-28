import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
)
import matplotlib.pyplot as plt



def run(data_file:str) -> dict:
    """Run logistic regression to extract important features from data_file
    
    Args:
        - data_file (str) : path to the data file, must contains a 'LABEL' columns

    Returns:
        - (dict) : results of the logistic regression, contains the following keys:
                    - cross-validation_acc_scores':cv_scores,
                    - cross-validation_acc_mean':np.mean(cv_scores),
                    - acc
                    - auc
                    - log_loss
                    - classification_report
                    - selected_features
    """

    # load data
    df = pd.read_csv(data_file)
    X = df.drop(columns=["LABEL"])
    nb = 0
    label_to_nb = {}
    for label in set(df['LABEL']):
        label_to_nb[label] = nb
        nb+=1
    y = df["LABEL"].map(label_to_nb)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the model pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Standardize features
        ("logreg", LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000))  # L1 regularization
    ])

    # # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    # Train the model on the training set
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # Predicted probabilities for positive class

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    log_loss_value = log_loss(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=list(label_to_nb.keys()))
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance using SelectFromModel
    selector = SelectFromModel(pipeline.named_steps["logreg"], prefit=True, threshold="mean")
    selected_features = X.columns[selector.get_support()]

    # craft results
    results = {
        'cross-validation_acc_scores':cv_scores,
        'cross-validation_acc_mean':np.mean(cv_scores),
        'acc':accuracy,
        'auc':roc_auc,
        'log_loss':log_loss,
        'classification_report':report,
        'selected_features':selected_features
    }

    # return computed results
    return results




if __name__ == "__main__":

    # Load the data
    data_file = "data/toy_data.csv"
    s = run(data_file)
