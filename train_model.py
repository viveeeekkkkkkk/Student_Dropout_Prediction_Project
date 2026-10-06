import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# STUDENT DROPOUT PREDICTION
# COMPLETE ML PIPELINE
# ============================================================

print("=" * 75)
print("STUDENT DROPOUT PREDICTION - COMPLETE ML PIPELINE")
print("=" * 75)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1] LOADING DATASET")
print("-" * 75)

df = pd.read_excel("student_dataset.xlsx")

print("Dataset loaded successfully.")
print(f"Total records : {len(df)}")
print(f"Total columns : {len(df.columns)}")

print("\nDataset columns:")
print(list(df.columns))


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

print("\n[2] DATA UNDERSTANDING")
print("-" * 75)

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 3. DATA QUALITY CHECK
# ============================================================

print("\n[3] DATA QUALITY CHECK")
print("-" * 75)

missing_values = df.isnull().sum()

print("\nMissing values by column:")
print(missing_values)

print(f"\nTotal missing values: {missing_values.sum()}")

duplicate_rows = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_rows}")


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\n[4] TARGET / CLASS DISTRIBUTION")
print("-" * 75)

label_counts = df["label"].value_counts().sort_index()

print("\nLabel distribution:")
print(label_counts)

print("\nLabel meaning:")
print("0 = Active Student")
print("1 = Dropout")

label_percentages = (
    df["label"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("\nLabel percentages:")

for label, percentage in label_percentages.items():

    if label == 0:
        name = "Active Student"
    else:
        name = "Dropout"

    print(f"{name}: {percentage:.2f}%")


# ============================================================
# 5. FEATURE SELECTION
# ============================================================

print("\n[5] FEATURE SELECTION")
print("-" * 75)

features = [
    "age",
    "exam_season",
    "courses_enrolled",
    "completed_assignments",
    "completion_rate",
    "login_frequency",
    "last_activity_days_ago",
    "forum_posts_count"
]

target = "label"

X = df[features].copy()
y = df[target].copy()

print("\nSelected features:")

for feature in features:
    print(" -", feature)

print("\nTarget:", target)

print("\nStudent ID is excluded because it is an identifier,")
print("not a meaningful predictive feature.")


# ============================================================
# 6. FEATURE RANGE CHECK
# ============================================================

print("\n[6] FEATURE RANGE / OUTLIER CHECK")
print("-" * 75)

print("\nNumerical feature summary:")

print(
    X.describe().round(3).to_string()
)

print("\nThis step is used to identify unusual or impossible values")
print("before model training.")


# ============================================================
# 7. EDA - CLASS DISTRIBUTION
# ============================================================

print("\n[7] EXPLORATORY DATA ANALYSIS")
print("-" * 75)

plt.figure(figsize=(7, 5))

sns.countplot(
    x=df["label"]
)

plt.title("Student Class Distribution")

plt.xlabel("Label")

plt.ylabel("Number of Students")

plt.xticks(
    [0, 1],
    ["Active (0)", "Dropout (1)"]
)

plt.tight_layout()

plt.savefig(
    "eda_class_distribution.png",
    dpi=300
)

plt.close()

print("Saved: eda_class_distribution.png")


# ============================================================
# 8. EDA - FEATURE DISTRIBUTIONS
# ============================================================

for feature in features:

    plt.figure(figsize=(7, 5))

    sns.histplot(
        data=df,
        x=feature,
        hue="label",
        kde=True,
        bins=20
    )

    plt.title(
        f"{feature} Distribution by Student Status"
    )

    plt.tight_layout()

    filename = (
        "eda_"
        + feature
        + ".png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

print("Feature distribution plots generated.")


# ============================================================
# 9. EDA - CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

correlation = df[
    features + [target]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "eda_correlation_heatmap.png",
    dpi=300
)

plt.close()

print("Saved: eda_correlation_heatmap.png")


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

print("\n[8] TRAIN / TEST SPLIT")
print("-" * 75)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records : {len(X_train)}")
print(f"Testing records  : {len(X_test)}")

print("\nThe test set will remain untouched until final evaluation.")


# ============================================================
# 11. DEFINE MACHINE LEARNING MODELS
# ============================================================

print("\n[9] DEFINING MACHINE LEARNING MODELS")
print("-" * 75)

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=9,
        random_state=42
    ),

    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
}

print("\nModels included:")

for name in models:
    print(" -", name)


# ============================================================
# 12. CROSS-VALIDATION
# ============================================================

print("\n[10] 5-FOLD STRATIFIED CROSS-VALIDATION")
print("-" * 75)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

cv_results = []

for name, model in models.items():

    print(f"\nCross-validating {name}...")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results.append({

        "Model": name,

        "CV Accuracy": scores[
            "test_accuracy"
        ].mean(),

        "CV Precision": scores[
            "test_precision"
        ].mean(),

        "CV Recall": scores[
            "test_recall"
        ].mean(),

        "CV F1": scores[
            "test_f1"
        ].mean(),

        "CV ROC-AUC": scores[
            "test_roc_auc"
        ].mean(),

        "CV PR-AUC": scores[
            "test_pr_auc"
        ].mean()
    })


# ============================================================
# 13. CROSS-VALIDATION COMPARISON
# ============================================================

cv_results_df = pd.DataFrame(
    cv_results
)

print("\n" + "=" * 75)
print("CROSS-VALIDATION RESULTS")
print("=" * 75)

display_cv = cv_results_df.copy()

for column in display_cv.columns[1:]:

    display_cv[column] = (
        display_cv[column] * 100
    ).round(2)

print(
    display_cv.to_string(index=False)
)


# ============================================================
# 14. SELECT MODEL USING VALIDATION EVIDENCE
# ============================================================

best_model_name = cv_results_df.loc[
    cv_results_df["CV F1"].idxmax(),
    "Model"
]

best_model = models[
    best_model_name
]

best_cv_row = cv_results_df[
    cv_results_df["Model"] == best_model_name
].iloc[0]


print("\n" + "=" * 75)
print("MODEL SELECTED USING CROSS-VALIDATION")
print("=" * 75)

print(
    f"Selected model : {best_model_name}"
)

print(
    f"CV Accuracy    : "
    f"{best_cv_row['CV Accuracy'] * 100:.2f}%"
)

print(
    f"CV Precision   : "
    f"{best_cv_row['CV Precision'] * 100:.2f}%"
)

print(
    f"CV Recall      : "
    f"{best_cv_row['CV Recall'] * 100:.2f}%"
)

print(
    f"CV F1          : "
    f"{best_cv_row['CV F1'] * 100:.2f}%"
)

print(
    f"CV ROC-AUC     : "
    f"{best_cv_row['CV ROC-AUC'] * 100:.2f}%"
)

print(
    f"CV PR-AUC      : "
    f"{best_cv_row['CV PR-AUC'] * 100:.2f}%"
)


# ============================================================
# 15. FINAL TRAINING
# ============================================================

print("\n[11] FINAL MODEL TRAINING")
print("-" * 75)

best_model.fit(
    X_train,
    y_train
)

print(
    "Final model trained using the complete training dataset."
)


# ============================================================
# 16. FINAL TEST EVALUATION
# ============================================================

print("\n[12] FINAL TEST EVALUATION")
print("-" * 75)

y_pred = best_model.predict(
    X_test
)

y_probability = best_model.predict_proba(
    X_test
)[:, 1]


final_accuracy = accuracy_score(
    y_test,
    y_pred
)

final_precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

final_roc_auc = roc_auc_score(
    y_test,
    y_probability
)

final_pr_auc = average_precision_score(
    y_test,
    y_probability
)


print("\nFINAL TEST RESULTS")

print(
    f"Accuracy  : {final_accuracy * 100:.2f}%"
)

print(
    f"Precision : {final_precision * 100:.2f}%"
)

print(
    f"Recall    : {final_recall * 100:.2f}%"
)

print(
    f"F1 Score  : {final_f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {final_roc_auc * 100:.2f}%"
)

print(
    f"PR-AUC    : {final_pr_auc * 100:.2f}%"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "ACTIVE (0)",
            "DROPOUT (1)"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm)

print("\nConfusion Matrix Details:")

print(
    "True Negative  (TN):",
    cm[0][0]
)

print(
    "False Positive (FP):",
    cm[0][1]
)

print(
    "False Negative (FN):",
    cm[1][0]
)

print(
    "True Positive  (TP):",
    cm[1][1]
)


# ============================================================
# 19. SAVE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=[
        "Active (0)",
        "Dropout (1)"
    ],
    yticklabels=[
        "Active (0)",
        "Dropout (1)"
    ]
)

plt.title(
    "Confusion Matrix - " + best_model_name
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()

print(
    "\nConfusion matrix saved as: "
    "confusion_matrix.png"
)


# ============================================================
# 20. FEATURE IMPORTANCE / EXPLAINABILITY
# ============================================================

print("\n[13] FEATURE IMPORTANCE / EXPLAINABILITY")
print("-" * 75)

feature_importance = None

# Tree-based models
if hasattr(
    best_model,
    "feature_importances_"
):

    feature_importance = (
        best_model.feature_importances_
    )

# Pipeline containing a tree model
elif hasattr(
    best_model,
    "named_steps"
):

    final_estimator = (
        best_model.named_steps["model"]
    )

    if hasattr(
        final_estimator,
        "feature_importances_"
    ):

        feature_importance = (
            final_estimator.feature_importances_
        )

    elif hasattr(
        final_estimator,
        "coef_"
    ):

        feature_importance = np.abs(
            final_estimator.coef_[0]
        )


if feature_importance is not None:

    importance_df = pd.DataFrame({

        "Feature": features,

        "Importance": feature_importance

    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nFeature importance:")

    print(
        importance_df.to_string(
            index=False
        )
    )

    importance_df.to_csv(
        "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Feature Importance - "
        + best_model_name
    )

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png",
        dpi=300
    )

    plt.close()

    print(
        "\nSaved: feature_importance.csv"
    )

    print(
        "Saved: feature_importance.png"
    )

else:

    print(
        "Feature importance is not directly available "
        "for this model."
    )


# ============================================================
# 21. MODEL COMPARISON FILE
# ============================================================

comparison_df = cv_results_df.copy()

comparison_df[
    "Final Test Accuracy"
] = np.nan

comparison_df[
    "Final Test Precision"
] = np.nan

comparison_df[
    "Final Test Recall"
] = np.nan

comparison_df[
    "Final Test F1"
] = np.nan

comparison_df[
    "Final Test ROC-AUC"
] = np.nan

comparison_df[
    "Final Test PR-AUC"
] = np.nan


selected_index = comparison_df.index[
    comparison_df["Model"]
    == best_model_name
][0]


comparison_df.loc[
    selected_index,
    "Final Test Accuracy"
] = final_accuracy

comparison_df.loc[
    selected_index,
    "Final Test Precision"
] = final_precision

comparison_df.loc[
    selected_index,
    "Final Test Recall"
] = final_recall

comparison_df.loc[
    selected_index,
    "Final Test F1"
] = final_f1

comparison_df.loc[
    selected_index,
    "Final Test ROC-AUC"
] = final_roc_auc

comparison_df.loc[
    selected_index,
    "Final Test PR-AUC"
] = final_pr_auc


comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)

print(
    "\nModel comparison saved as: "
    "model_comparison.csv"
)


# ============================================================
# 22. SAVE FINAL MODEL
# ============================================================

print("\n[14] SAVING FINAL MODEL")
print("-" * 75)

bundle = {

    "model": best_model,

    "features": features,

    "model_name": best_model_name,

    "cv_f1": float(
        best_cv_row["CV F1"]
    ),

    "accuracy": float(
        final_accuracy
    ),

    "precision": float(
        final_precision
    ),

    "recall": float(
        final_recall
    ),

    "f1_score": float(
        final_f1
    ),

    "roc_auc": float(
        final_roc_auc
    ),

    "pr_auc": float(
        final_pr_auc
    )
}


joblib.dump(
    bundle,
    "model.pkl"
)


print(
    "Model saved successfully: model.pkl"
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PROJECT ML PIPELINE COMPLETED")
print("=" * 75)

print(
    f"\nFinal selected model: "
    f"{best_model_name}"
)

print(
    f"Final test accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Final test F1: "
    f"{final_f1 * 100:.2f}%"
)

print(
    f"Final test ROC-AUC: "
    f"{final_roc_auc * 100:.2f}%"
)

print(
    f"Final test PR-AUC: "
    f"{final_pr_auc * 100:.2f}%"
)

print("\nGenerated files:")

print(" - model.pkl")
print(" - model_comparison.csv")
print(" - confusion_matrix.png")
print(" - feature_importance.csv")
print(" - feature_importance.png")
print(" - eda_class_distribution.png")
print(" - eda_correlation_heatmap.png")
print(" - EDA feature distribution plots")

print("\nTraining pipeline completed successfully.")