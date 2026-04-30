from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test, threshold=0.3):
    print("Evaluating model...")

    y_prob = model.predict_proba(x_test)[:,1]
    y_pred = (y_prob > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy