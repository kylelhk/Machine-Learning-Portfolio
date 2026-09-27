# Applied Machine Learning & Predictive Analytics Portfolio

This repository contains a collection of machine learning projects developed in Python using **Scikit-Learn, Pandas, NumPy, and Seaborn**. The projects cover end-to-end ML workflows—including exploratory data analysis (EDA), statistical preprocessing, feature engineering, model selection, hyperparameter optimisation, dimensionality reduction, and unsupervised clustering.

---

## 📁 Projects Overview

| File Name | Dataset | Key Techniques & Models | Primary Focus |
| :--- | :--- | :--- | :--- |
| `fashion_mnist_binary_classification.ipynb` | Fashion-MNIST | Logistic Regression, Gradient Descent, k-NN, L2 Regularisation, Grid Search | Custom optimisation algorithm implementation, diagnostic evaluation, cross-validation |
| `breast_cancer_decision_trees.ipynb` | Breast Cancer Wisconsin | Decision Trees, Random Forests, Feature Importance, Cost-Complexity Pruning | Overfitting diagnostics, feature trimming, tree interpretability, ensemble learning |
| `california_housing_regression_clustering.ipynb` | California Housing (1990 Census) | Lasso, Ridge, Decision Tree Regressor, PCA, Hierarchical & k-Means Clustering | Feature engineering, dimensionality reduction, regression modeling, unsupervised grouping |

---

## 💻 Tech Stack & Dependencies

* **Language:** Python
* **Core Libraries:** `scikit-learn`, `pandas`, `numpy`
* **Visualisation:** `matplotlib`, `seaborn`
* **Environment:** Jupyter Notebook / VS Code
---

## 🛠️ Detailed Project Breakdowns

### 1. Fashion-MNIST Binary Classification (`fashion_mnist_binary_classification.ipynb`)
* **Objective:** Classify footwear images (*Sneakers* vs. *Sandals*) from 784-dimensional pixel arrays ($28\times28$ grayscale).
* **Technical Highlights:**
  * Implemented **Batch Gradient Descent from scratch** to optimise Logistic Regression cost functions.
  * Evaluated regularisation hyperparameter $C$ via `GridSearchCV` with **10-fold cross-validation**.
  * Tuned decision thresholds using **Precision-Recall curve analysis** to optimise model generalisation.
  * Benchmark performance against **k-Nearest Neighbours (k-NN)** Euclidean distance models across multiple values of $k$.
  * Analysed error cases by visualising **False Positives / Negatives** and mapping learned weight coefficients back to 2D image matrices.

### 2. Breast Cancer Diagnostic Classification (`breast_cancer_decision_trees.ipynb`)
* **Objective:** Predict malignant vs. benign tumors based on nuclear feature characteristics.
* **Technical Highlights:**
  * Conducted correlation analysis and feature selection, dropping highly collinear features ($\vert{}r\vert{} > 0.97$) to mitigate multicollinearity.
  * Evaluated Decision Tree baseline performance across multiple random splits and varying training set sizes ($50\%$ to $90\%$).
  * Performed hyperparameter tuning (`max_depth`, `min_samples_split`, `min_samples_leaf`) using **10-fold cross-validation**.
  * Trimmed feature space by extracting **Feature Importances** ($>1\%$ threshold) and re-evaluating model metrics.
  * Built and benchmarked **Random Forest ensembles** against standalone Decision Trees.

### 3. California Housing Price Prediction & Clustering (`california_housing_regression_clustering.ipynb`)
* **Objective:** Model district-level median house values and uncover geographic / demographic clusters.
* **Technical Highlights:**
  * Engineered domain features including `meanRooms`, `meanBedrooms`, and `meanOccupation`.
  * Applied **Lasso ($\mathbf{L_1}$)** and **Ridge ($\mathbf{L_2}$)** regularisation to handle multicollinearity and feature selection.
  * Applied **Principal Component Analysis (PCA)** to reduce feature dimensionality while preserving $\ge 90\%$ explained variance.
  * Performed **Hierarchical Agglomerative Clustering** (Average Linkage, Euclidean) and **k-Means Clustering** ($k=4$).
  * Evaluated optimal cluster counts ($k=2$ to $20$) using **Silhouette Coefficient analysis** mapped on 2D PCA projections.
