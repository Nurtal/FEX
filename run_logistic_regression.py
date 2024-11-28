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



def run(data_file):
    """ """

    df = pd.read_csv(data_file)

    # Separate features and target
    X = df.drop(columns=["Group"])  # Assuming 'Group' is the target column
    y = df["Group"].map({"A": 0, "B": 1})  # Convert 'A' and 'B' to binary labels

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the model pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Standardize features
        ("logreg", LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000))  # L1 regularization
    ])

    # Cross-validation setup
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
    report = classification_report(y_test, y_pred, target_names=["Group A", "Group B"])
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance using SelectFromModel
    selector = SelectFromModel(pipeline.named_steps["logreg"], prefit=True, threshold="mean")
    selected_features = X.columns[selector.get_support()]

    # Display results
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Log Loss: {log_loss_value:.4f}")
    print("Classification Report:\n", report)
    print(f"Selected Features: {list(selected_features)}")

    # Plot confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Group A", "Group B"]).plot()




if __name__ == "__main__":

    # Load the data
    data_file = "data/toy_data.csv"
    run(data_file)
