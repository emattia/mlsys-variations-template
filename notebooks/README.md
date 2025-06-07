# ML Research & Experimentation Notebooks

This directory contains Jupyter notebooks for data science research, ML experimentation, and exploratory analysis. These notebooks serve as the interactive environment for data scientists and researchers to explore data, develop models, and validate hypotheses.

## 🎯 **Purpose**

The notebooks directory enables **data scientists and researchers** to:

- **🔍 Explore datasets** with interactive visualizations and statistical analysis
- **🧪 Experiment with ML models** in a rapid prototyping environment
- **📊 Create research reports** with embedded code, plots, and findings
- **🔬 Validate hypotheses** through systematic experimentation
- **📈 Develop visualizations** for stakeholder communication
- **🔄 Prototype workflows** before production implementation

## 🏗️ **Notebook Architecture**

```
notebooks/
├── 📊 Data Exploration
│   ├── 01_exploratory_data_analysis.ipynb    # Initial data investigation
│   ├── 02_data_quality_assessment.ipynb      # Data quality checks
│   └── 03_statistical_analysis.ipynb         # Statistical summaries
│
├── 🤖 Model Development
│   ├── 10_feature_engineering.ipynb          # Feature creation & selection
│   ├── 11_baseline_models.ipynb              # Simple baseline comparisons
│   ├── 12_advanced_modeling.ipynb            # Complex model experiments
│   └── 13_model_interpretation.ipynb         # SHAP, LIME, feature importance
│
├── 🧪 Experiments
│   ├── 20_experiment_design.ipynb            # Experimental setup
│   ├── 21_hyperparameter_tuning.ipynb       # Model optimization
│   ├── 22_cross_validation.ipynb            # Validation strategies
│   └── 23_ensemble_methods.ipynb            # Model combination
│
├── 📈 Analysis & Reporting
│   ├── 30_performance_analysis.ipynb         # Model evaluation
│   ├── 31_business_impact.ipynb             # Business metrics analysis
│   ├── 32_comparative_study.ipynb           # Model comparisons
│   └── 33_final_report.ipynb                # Executive summary
│
└── 🔧 Utilities & Templates
    ├── 99_notebook_template.ipynb           # Standard notebook template
    └── utils/                               # Shared notebook utilities
        ├── plotting.py                      # Custom plotting functions
        └── analysis.py                      # Analysis helpers
```

---

## 🚀 **Quick Start**

### **Launch Jupyter Environment**

```bash
# Start JupyterLab with project environment
make jupyter

# Alternative: Direct launch
uv run jupyter lab

# Access at: http://localhost:8888
```

### **Create New Analysis**

```bash
# Copy the template for new experiments
cp notebooks/99_notebook_template.ipynb notebooks/50_my_experiment.ipynb

# Start with data exploration template
cp notebooks/01_exploratory_data_analysis.ipynb notebooks/51_my_data_exploration.ipynb
```

### **Connect to Production Data**

```python
# Example: Load data from the ML pipeline
import sys
sys.path.append('..')

from src.data.loading import load_data
from src.config import load_config

# Load configuration and data
config = load_config()
data = load_data("data/processed/features.parquet")

print(f"Dataset shape: {data.shape}")
```

---

## 📊 **Data Exploration Workflows**

### **Interactive Data Analysis**

```python
# 01_exploratory_data_analysis.ipynb - Key patterns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from ydata_profiling import ProfileReport

# Load and profile data
df = pd.read_parquet("../data/processed/features.parquet")

# Generate automated EDA report
profile = ProfileReport(df, title="Dataset Profile Report")
profile.to_notebook_iframe()

# Interactive visualizations
fig = px.scatter_matrix(df.select_dtypes(include=[np.number]).sample(1000),
                       title="Feature Correlation Matrix")
fig.show()
```

### **Statistical Analysis Templates**

```python
# 03_statistical_analysis.ipynb - Analysis patterns
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Hypothesis testing template
def perform_statistical_tests(data, target_col):
    """Comprehensive statistical analysis."""
    results = {}

    # Normality tests
    for col in data.select_dtypes(include=[np.number]).columns:
        stat, p_value = stats.shapiro(data[col].dropna())
        results[f"{col}_normality"] = {"statistic": stat, "p_value": p_value}

    # Correlation analysis
    correlation_matrix = data.corr()

    # Feature importance via mutual information
    from sklearn.feature_selection import mutual_info_regression
    mi_scores = mutual_info_regression(data.drop(columns=[target_col]), data[target_col])

    return {
        "normality_tests": results,
        "correlations": correlation_matrix,
        "mutual_information": dict(zip(data.columns[:-1], mi_scores))
    }

# Usage
stats_results = perform_statistical_tests(df, 'target')
```

---

## 🤖 **Model Development Patterns**

### **Rapid Prototyping Template**

```python
# 11_baseline_models.ipynb - Quick model comparison
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Quick model comparison pipeline
def compare_baseline_models(X, y, cv=5):
    """Compare multiple baseline models quickly."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42)
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }

    return results

# Usage
X = df.drop(columns=['target'])
y = df['target']
baseline_results = compare_baseline_models(X, y)

# Visualize results
import matplotlib.pyplot as plt
models = list(baseline_results.keys())
means = [baseline_results[m]['mean_accuracy'] for m in models]
stds = [baseline_results[m]['std_accuracy'] for m in models]

plt.figure(figsize=(10, 6))
plt.bar(models, means, yerr=stds, capsize=5)
plt.title('Baseline Model Comparison')
plt.ylabel('Cross-Validation Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### **Advanced Experimentation**

```python
# 12_advanced_modeling.ipynb - Deep learning experiments
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class NeuralNetworkExperiment:
    """Template for neural network experiments."""

    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1):
        self.model = self._build_model(input_dim, hidden_dims, output_dim)
        self.scaler = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': []}

    def _build_model(self, input_dim, hidden_dims, output_dim):
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

# Usage example
experiment = NeuralNetworkExperiment(input_dim=X.shape[1])
```

---

## 🧪 **Experimentation Framework**

### **Experiment Tracking**

```python
# 20_experiment_design.ipynb - MLflow integration
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Initialize experiment tracking
mlflow.set_experiment("model_comparison_experiment")

def run_experiment(model, X_train, y_train, X_test, y_test, params=None):
    """Run and track ML experiment."""
    with mlflow.start_run():
        # Log parameters
        if params:
            mlflow.log_params(params)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Create and log plots
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        return model, {"train_acc": train_score, "test_acc": test_score}

# Usage
model = RandomForestClassifier(n_estimators=100, random_state=42)
trained_model, metrics = run_experiment(model, X_train, y_train, X_test, y_test)
```

### **Hyperparameter Optimization**

```python
# 21_hyperparameter_tuning.ipynb - Optuna integration
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create model with suggested parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Evaluate with cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.4f}")

# Visualize optimization
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

---

## 📈 **Advanced Analysis & Interpretation**

### **Model Interpretation**

```python
# 13_model_interpretation.ipynb - SHAP analysis
import shap
from sklearn.inspection import permutation_importance

class ModelInterpreter:
    """Comprehensive model interpretation toolkit."""

    def __init__(self, model, X_train, X_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names

    def shap_analysis(self, sample_size=1000):
        """Generate SHAP explanations."""
        # Initialize explainer
        explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(self.X_test[:sample_size])

        # Summary plot
        shap.summary_plot(shap_values, self.X_test[:sample_size],
                         feature_names=self.feature_names)

        # Feature importance
        shap.summary_plot(shap_values, self.X_test[:sample_size],
                         feature_names=self.feature_names, plot_type="bar")

        return shap_values

    def permutation_importance_analysis(self):
        """Calculate permutation feature importance."""
        perm_importance = permutation_importance(
            self.model, self.X_test, y_test,
            n_repeats=10, random_state=42
        )

        # Create DataFrame for easier analysis
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        # Visualize
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15], importance_df['importance_mean'][:15])
        plt.xlabel('Permutation Importance')
        plt.title('Top 15 Features by Permutation Importance')
        plt.tight_layout()
        plt.show()

        return importance_df

# Usage
interpreter = ModelInterpreter(trained_model, X_train, X_test, X.columns.tolist())
shap_values = interpreter.shap_analysis()
importance_df = interpreter.permutation_importance_analysis()
```

### **Business Impact Analysis**

```python
# 31_business_impact.ipynb - ROI calculations
class BusinessImpactAnalyzer:
    """Calculate business impact of ML models."""

    def __init__(self, true_positives, false_positives, true_negatives, false_negatives):
        self.tp = true_positives
        self.fp = false_positives
        self.tn = true_negatives
        self.fn = false_negatives

    def calculate_roi(self, benefit_per_tp=100, cost_per_fp=50, cost_per_fn=200):
        """Calculate ROI based on business costs/benefits."""
        total_benefit = self.tp * benefit_per_tp
        total_cost = (self.fp * cost_per_fp) + (self.fn * cost_per_fn)

        roi = (total_benefit - total_cost) / total_cost if total_cost > 0 else 0

        return {
            'total_benefit': total_benefit,
            'total_cost': total_cost,
            'net_benefit': total_benefit - total_cost,
            'roi_percentage': roi * 100
        }

    def threshold_analysis(self, y_true, y_proba, benefit_per_tp=100,
                          cost_per_fp=50, cost_per_fn=200):
        """Find optimal threshold for business metrics."""
        from sklearn.metrics import confusion_matrix

        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            analyzer = BusinessImpactAnalyzer(tp, fp, tn, fn)
            roi_metrics = analyzer.calculate_roi(benefit_per_tp, cost_per_fp, cost_per_fn)

            results.append({
                'threshold': threshold,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                **roi_metrics
            })

        return pd.DataFrame(results)

# Usage
y_proba = trained_model.predict_proba(X_test)[:, 1]
impact_analyzer = BusinessImpactAnalyzer(0, 0, 0, 0)  # Will be calculated in threshold analysis
threshold_results = impact_analyzer.threshold_analysis(y_test, y_proba)

# Find optimal threshold
optimal_threshold = threshold_results.loc[threshold_results['roi_percentage'].idxmax(), 'threshold']
print(f"Optimal threshold for ROI: {optimal_threshold:.3f}")
```

---

## 🛠️ **Development Tools & Best Practices**

### **Notebook Template Structure**

```python
# 99_notebook_template.ipynb - Standard structure
"""
Notebook: [Title]
Author: [Your Name]
Date: [Date]
Purpose: [Brief description of analysis goals]
Status: [Draft/In Progress/Complete]
"""

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

# Standard data science stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Project modules
import sys
sys.path.append('..')
from src.config import load_config
from src.data.loading import load_data

# Configuration
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)
np.random.seed(42)

config = load_config()

# =============================================================================
# 2. DATA LOADING & INITIAL EXPLORATION
# =============================================================================
# [Your data loading code here]

# =============================================================================
# 3. ANALYSIS & EXPERIMENTATION
# =============================================================================
# [Main analysis code here]

# =============================================================================
# 4. RESULTS & CONCLUSIONS
# =============================================================================
# [Summary of findings]

# =============================================================================
# 5. NEXT STEPS
# =============================================================================
# [Action items and future work]
```

### **Notebook Quality Checklist**

```markdown
## Pre-Commit Checklist

- [ ] **Clear title and purpose** in first cell
- [ ] **All imports** organized in setup section
- [ ] **Markdown explanations** between code sections
- [ ] **Meaningful variable names** throughout
- [ ] **Plots have titles and labels**
- [ ] **Key findings summarized**
- [ ] **Outputs cleared** before commit
- [ ] **No hardcoded paths** - use config
- [ ] **Reproducible** - set random seeds
- [ ] **Performance checked** - runtime under 5 minutes
```

### **Utility Functions**

```python
# notebooks/utils/plotting.py - Shared plotting utilities
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def setup_plotting():
    """Configure plotting defaults for notebooks."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Matplotlib settings
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16

def quick_eda_plots(df, target_col=None):
    """Generate standard EDA plots quickly."""
    # Distribution plots
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Correlation heatmap
        sns.heatmap(df[numeric_cols].corr(), annot=True, ax=axes[0,0])
        axes[0,0].set_title('Correlation Matrix')

        # Distribution of target (if provided)
        if target_col and target_col in df.columns:
            df[target_col].hist(bins=30, ax=axes[0,1])
            axes[0,1].set_title(f'Distribution of {target_col}')

        # Missing values
        missing_data = df.isnull().sum()
        missing_data[missing_data > 0].plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Missing Values by Column')

        # Data types
        dtype_counts = df.dtypes.value_counts()
        dtype_counts.plot(kind='pie', ax=axes[1,1])
        axes[1,1].set_title('Data Types Distribution')

        plt.tight_layout()
        plt.show()

def plot_model_performance(y_true, y_pred, y_proba=None):
    """Comprehensive model performance visualization."""
    from sklearn.metrics import confusion_matrix, roc_curve, auc

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0])
    axes[0].set_title('Confusion Matrix')

    # ROC Curve (if probabilities provided)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()

    # Feature importance (if available)
    if hasattr(y_pred, 'feature_importances_'):
        feature_imp = pd.Series(y_pred.feature_importances_).sort_values(ascending=False)
        feature_imp.head(10).plot(kind='barh', ax=axes[2])
        axes[2].set_title('Top 10 Feature Importances')

    plt.tight_layout()
    plt.show()
```

---

## 🔧 **Configuration & Environment**

### **Jupyter Extensions**

```bash
# Install useful Jupyter extensions
uv pip install jupyterlab-git
uv pip install jupyterlab-lsp
uv pip install jupyterlab-code-formatter
uv pip install nbstripout

# Configure git to strip notebook outputs
nbstripout --install --attributes .gitattributes
```

### **Environment Variables**

```python
# Set environment for notebook development
import os
os.environ['PYTHONPATH'] = '/path/to/project'
os.environ['JUPYTER_CONFIG_DIR'] = '/path/to/jupyter/config'

# GPU configuration (if available)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

---

## 📚 **Additional Resources**

- **[Jupyter Documentation](https://jupyter.org/documentation)** - Official Jupyter guides
- **[Data Science Best Practices](../docs/data_science_guide.md)** - Project-specific guidelines
- **[Visualization Gallery](../docs/visualization_examples.md)** - Plot examples and templates
- **[MLflow Integration](../docs/experiment_tracking.md)** - Experiment tracking setup
- **[Model Deployment](../workflows/README.md)** - Moving from notebooks to production

---

## 🤝 **Contributing**

1. **Use the template** for all new notebooks
2. **Follow naming conventions**: `##_descriptive_name.ipynb`
3. **Document thoroughly** with markdown cells
4. **Clear outputs** before committing
5. **Add to this README** when creating reusable patterns
6. **Test reproducibility** - can others run your notebook?

Remember: Notebooks are for exploration and prototyping. Move production code to `src/` and workflows to `workflows/`!
