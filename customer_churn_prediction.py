# =============================================================================
# Customer Churn Prediction using Machine Learning
# Internship Project - UpSkill Campus & UniConverge Technologies Pvt Ltd
# Domain: Data Science and Machine Learning
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: GENERATE / LOAD DATASET
# =============================================================================
# Using synthetically generated dataset that mimics real telecom churn data.
# In production, replace this block with: df = pd.read_csv("your_dataset.csv")

np.random.seed(42)
n = 1000

def generate_churn_dataset(n_samples=1000):
    """Generate a realistic telecom customer churn dataset."""
    age            = np.random.randint(18, 70, n_samples)
    tenure         = np.random.randint(1, 72, n_samples)          # months
    monthly_charge = np.round(np.random.uniform(20, 120, n_samples), 2)
    total_charges  = np.round(monthly_charge * tenure + np.random.normal(0, 50, n_samples), 2)
    num_products   = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    contract_type  = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                      n_samples, p=[0.55, 0.25, 0.20])
    internet_svc   = np.random.choice(['DSL', 'Fiber Optic', 'No'], n_samples,
                                      p=[0.35, 0.45, 0.20])
    payment_method = np.random.choice(
        ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'],
        n_samples, p=[0.35, 0.20, 0.25, 0.20]
    )
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])

    # Churn probability based on real-world correlations
    churn_prob = (
        0.05
        + 0.25 * (contract_type == 'Month-to-Month')
        - 0.10 * (contract_type == 'Two Year')
        + 0.15 * (internet_svc == 'Fiber Optic')
        + 0.10 * (payment_method == 'Electronic Check')
        + 0.10 * senior_citizen
        - 0.003 * tenure
        + 0.002 * monthly_charge
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.90)
    churn      = np.random.binomial(1, churn_prob)

    return pd.DataFrame({
        'Age'            : age,
        'Tenure'         : tenure,
        'MonthlyCharges' : monthly_charge,
        'TotalCharges'   : total_charges,
        'NumProducts'    : num_products,
        'ContractType'   : contract_type,
        'InternetService': internet_svc,
        'PaymentMethod'  : payment_method,
        'SeniorCitizen'  : senior_citizen,
        'Churn'          : churn
    })

print("=" * 60)
print("   Customer Churn Prediction — ML Pipeline")
print("   UpSkill Campus | Data Science & ML Internship")
print("=" * 60)

df = generate_churn_dataset(n)
print(f"\n[1] Dataset loaded  →  Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n[2] Exploratory Data Analysis")
print("-" * 40)
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print(f"\nChurn Distribution:\n{df['Churn'].value_counts()}")
print(f"Churn Rate: {df['Churn'].mean() * 100:.2f}%")
print(f"Missing Values:\n{df.isnull().sum()}")

# Visualisation — save plots as PNG files
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Customer Churn — EDA Dashboard", fontsize=16, fontweight='bold')

# 1. Churn Distribution
churn_counts = df['Churn'].value_counts()
axes[0, 0].bar(['No Churn', 'Churned'], churn_counts.values,
                color=['#2ecc71', '#e74c3c'], edgecolor='white')
axes[0, 0].set_title('Churn Distribution')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(churn_counts.values):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# 2. Monthly Charges vs Churn
churned     = df[df['Churn'] == 1]['MonthlyCharges']
not_churned = df[df['Churn'] == 0]['MonthlyCharges']
axes[0, 1].hist(not_churned, alpha=0.7, label='No Churn', color='#2ecc71', bins=20)
axes[0, 1].hist(churned,     alpha=0.7, label='Churned',  color='#e74c3c', bins=20)
axes[0, 1].set_title('Monthly Charges vs Churn')
axes[0, 1].set_xlabel('Monthly Charges ($)')
axes[0, 1].legend()

# 3. Contract Type vs Churn
contract_churn = df.groupby('ContractType')['Churn'].mean() * 100
axes[0, 2].bar(contract_churn.index, contract_churn.values,
               color=['#3498db', '#9b59b6', '#e67e22'])
axes[0, 2].set_title('Churn Rate by Contract Type')
axes[0, 2].set_ylabel('Churn Rate (%)')
axes[0, 2].tick_params(axis='x', rotation=15)

# 4. Tenure Distribution
axes[1, 0].hist(df[df['Churn'] == 0]['Tenure'], alpha=0.7,
                label='No Churn', color='#2ecc71', bins=20)
axes[1, 0].hist(df[df['Churn'] == 1]['Tenure'], alpha=0.7,
                label='Churned',  color='#e74c3c', bins=20)
axes[1, 0].set_title('Tenure Distribution vs Churn')
axes[1, 0].set_xlabel('Tenure (Months)')
axes[1, 0].legend()

# 5. Internet Service vs Churn
internet_churn = df.groupby('InternetService')['Churn'].mean() * 100
axes[1, 1].bar(internet_churn.index, internet_churn.values,
               color=['#1abc9c', '#e74c3c', '#95a5a6'])
axes[1, 1].set_title('Churn Rate by Internet Service')
axes[1, 1].set_ylabel('Churn Rate (%)')

# 6. Correlation Heatmap (numeric cols)
numeric_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges',
                'NumProducts', 'SeniorCitizen', 'Churn']
corr = df[numeric_cols].corr()
im   = axes[1, 2].imshow(corr, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
axes[1, 2].set_xticks(range(len(numeric_cols)))
axes[1, 2].set_yticks(range(len(numeric_cols)))
axes[1, 2].set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=7)
axes[1, 2].set_yticklabels(numeric_cols, fontsize=7)
axes[1, 2].set_title('Correlation Heatmap')
plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("   → EDA dashboard saved as 'eda_dashboard.png'")

# =============================================================================
# STEP 3: DATA PREPROCESSING
# =============================================================================
print("\n[3] Data Preprocessing")
print("-" * 40)

df_model = df.copy()

# Encode categorical variables
le = LabelEncoder()
cat_cols = ['ContractType', 'InternetService', 'PaymentMethod']
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])
    print(f"   Encoded: {col}")

# Features & Target
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# Train-Test Split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")
print(f"   Features         : {X_train.shape[1]}")

# =============================================================================
# STEP 4: MODEL TRAINING
# =============================================================================
print("\n[4] Model Training & Evaluation")
print("-" * 40)

models = {
    'Logistic Regression'    : LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'          : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'      : GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv  = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()

    results[name] = {'model': model, 'accuracy': acc, 'auc': auc, 'cv_score': cv,
                     'y_pred': y_pred, 'y_proba': y_proba}

    print(f"\n  ── {name} ──")
    print(f"     Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"     ROC-AUC   : {auc:.4f}")
    print(f"     CV Score  : {cv:.4f}")
    print(f"\n     Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Churn','Churned'])}")

# =============================================================================
# STEP 5: BEST MODEL & VISUALISATIONS
# =============================================================================
best_name  = max(results, key=lambda k: results[k]['auc'])
best       = results[best_name]
best_model = best['model']

print(f"\n[5] Best Model Selected: {best_name}  (AUC = {best['auc']:.4f})")

# Confusion Matrix + ROC Curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Best Model Performance — {best_name}", fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, best['y_pred'])
im = ax1.imshow(cm, cmap='Blues')
ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
ax1.set_xticklabels(['No Churn', 'Churned'])
ax1.set_yticklabels(['No Churn', 'Churned'])
ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm[i, j], ha='center', va='center', fontsize=16, fontweight='bold',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')

# ROC Curve (all models)
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax2.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={res['auc']:.3f})")
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve Comparison'); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   → Model performance chart saved as 'model_performance.png'")

# Feature Importance (Random Forest)
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
colors = ['#e74c3c' if v > importances.mean() else '#3498db' for v in importances.values]
importances.plot(kind='barh', color=colors)
plt.title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   → Feature importance chart saved as 'feature_importance.png'")

# =============================================================================
# STEP 6: SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("   FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<30} {'Accuracy':>10} {'AUC':>8} {'CV Score':>10}")
print("-" * 60)
for name, res in results.items():
    print(f"{name:<30} {res['accuracy']:>10.4f} {res['auc']:>8.4f} {res['cv_score']:>10.4f}")
print("=" * 60)
print(f"\n✅ Best Performing Model : {best_name}")
print(f"   Accuracy             : {best['accuracy']*100:.2f}%")
print(f"   ROC-AUC Score        : {best['auc']:.4f}")
print("\n✅ Output files generated:")
print("   - eda_dashboard.png")
print("   - model_performance.png")
print("   - feature_importance.png")
print("\nProject complete! Upload all files to GitHub → upskillCampus repository.")
