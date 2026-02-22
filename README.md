# Data Generation using Modelling and Simulation for Machine Learning

## 📌 Objective

The objective of this assignment is to generate synthetic datasets using a simulation tool, analyze parameter bounds, perform multiple simulations, and compare different machine learning models to identify the best performing model.

---

## 🔧 Simulation Tool Used

We selected:

`sklearn.datasets.make_classification`

### Why this tool?
- Allows controlled synthetic dataset generation
- Supports adjustable parameters
- Suitable for classification problems
- Easy integration with ML models
- Works directly in Google Colab

---

## ⚙️ Parameter Analysis and Bounds

The following parameters were studied and bounded:

| Parameter        | Lower Bound | Upper Bound |
|------------------|------------|------------|
| n_samples        | 100        | 5000       |
| n_features       | 5          | 50         |
| n_informative    | Based on constraint | n_features - 1 |
| n_redundant      | 0          | n_features - n_informative |
| n_classes        | 2          | 5          |
| class_sep        | 0.5        | 3.0        |
| flip_y           | 0.0        | 0.3        |

### Important Constraint

The simulator requires:

n_classes × n_clusters_per_class ≤ 2^n_informative

To avoid constraint violations, we fixed:

n_clusters_per_class = 1

This ensures valid dataset generation for all simulations.

---

## 🔁 Step 1: Single Simulation Example

Random parameters were generated and passed to the simulator.  
Logistic Regression was trained and accuracy was recorded.

---

## 🔁 Step 2: 1000 Simulations

- Random parameters generated within defined bounds
- Dataset generated using simulation tool
- Logistic Regression trained
- Accuracy recorded
- Results stored in a Pandas DataFrame

Total simulations performed: **1000**

Each simulation stored:

- n_samples
- n_features
- n_informative
- n_redundant
- n_classes
- class_sep
- flip_y
- accuracy

---

## 🤖 Model Comparison

We compared the following Machine Learning models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes

Each model was trained on the same simulated dataset and evaluated using Accuracy.

---

## 📊 Results

| Model | Accuracy |
|-------|----------|
| SVM | 0.9167 |
| KNN | 0.9133 |
| Random Forest | 0.9017 |
| Logistic Regression | 0.8967 |
| Naive Bayes | 0.8967 |
| Decision Tree | 0.8250 |

---

## 🏆 Best Performing Model

**Support Vector Machine (SVM)**  
Accuracy: **91.67%**

---

## 📈 Observations

- Model performance depends heavily on dataset parameters.
- Higher class separation improves accuracy.
- Noise (flip_y) reduces model performance.
- SVM performed best among all tested models.

---

## 🧠 Conclusion

Synthetic data generation using modelling and simulation is highly useful for:

- Benchmarking ML models
- Testing robustness
- Studying parameter impact
- Generating scalable experimental datasets

This assignment successfully demonstrated:
- Controlled simulation
- Large-scale experiment (1000 runs)
- Model comparison and evaluation
- Performance analysis

---

## 🛠 Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Google Colab

---

## 📂 Repository Contents

- Jupyter Notebook (.ipynb)
- README.md
- Simulation and Model Comparison Code

---

## ✅ Final Outcome

Simulation-based data generation was successfully implemented and evaluated.  
Among all models tested, **SVM achieved the highest accuracy (91.67%)** and was identified as the best performing model.
