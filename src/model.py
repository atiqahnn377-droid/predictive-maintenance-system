from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(x_train, y_train, x_test, y_test):

    print("Training and comparing models...")

    # ===================================
    # RANDOM FOREST MODEL
    # ===================================
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    rf_model.fit(x_train, y_train)

    rf_pred = rf_model.predict(x_test)
    rf_acc = accuracy_score(y_test, rf_pred)


    # ===================================
    # SVM MODEL
    # ===================================
    svm_model = SVC(
        probability=True,
        random_state=42
    )

    svm_model.fit(x_train, y_train)

    svm_pred = svm_model.predict(x_test)
    svm_acc = accuracy_score(y_test, svm_pred)


    # ===================================
    # LOGISTIC REGRESSION MODEL
    # ===================================
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    lr_model.fit(x_train, y_train)

    lr_pred = lr_model.predict(x_test)
    lr_acc = accuracy_score(y_test, lr_pred)



    # ================================
    # PRINT MODEL COMPARISON RESULTS
    # ================================
    print("\n===== MODEL COMPARISON RESULTS =====")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")



    # ================================
    # STORE MODEL SCORES
    # ================================
    model_scores = {
        "Random Forest": rf_acc,
        "SVM": svm_acc,
        "Logistic Regression": lr_acc
    }

    
    # ================================
    # FIND BEST MODEL
    # ================================
    best_model_name = max(model_scores, key=model_scores.get)

    print(f"\nBest Model: {best_model_name} ({model_scores[best_model_name]:.4f})")



    # ================================
    # RETURN BEST MODEL
    # ================================
    if best_model_name == "Random Forest":
        return rf_model
    
    elif best_model_name =="SVM":
        return svm_model
    
    else:
        return lr_model
