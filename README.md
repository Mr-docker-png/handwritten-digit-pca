Handwritten Digit Recognition with PCA

A machine learning project that uses Principal Component Analysis (PCA) to reduce the dimensionality of handwritten digit images and Logistic Regression to classify the digits.

📌 Overview

The Scikit-learn Digits dataset contains 1,797 handwritten digit images. Each image is an 8×8 grayscale image, represented by 64 pixel features.

In this project, PCA is used to reduce the 64-dimensional feature space while preserving most of the information.

The reduced features are then used with Logistic Regression for digit classification.

Workflow
Digits Dataset
      ↓
Exploratory Data Analysis
      ↓
64 Pixel Features
      ↓
StandardScaler
      ↓
PCA
      ↓
31 Principal Components
      ↓
Logistic Regression
      ↓
Digit Prediction
      ↓
Model Evaluation
📊 Dataset

The project uses the built-in Scikit-learn Digits Dataset.

Property	Value
Samples	1,797
Original features	64
Image size	8 × 8
Classes	10
Classes	0–9
Pixel range	0–16

Each row contains 64 pixel values representing one 8×8 handwritten digit.

🧠 PCA

PCA was first applied without limiting the number of components to examine the explained variance.

The cumulative explained variance showed:

Components	Variance retained
10	58.87%
20	79.31%
25	85.13%
31	90.05%
40	95.08%
50	98.49%

We selected 90% explained variance.

PCA automatically selected:

64 → 31 components

Therefore, the feature space was reduced by 33 features while retaining approximately 90.06% of the variance.

⚙️ Pipeline

The final model uses a Scikit-learn Pipeline:

model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ("model", LogisticRegression(max_iter=2000))
])

This ensures that scaling and PCA are fitted only on the training data when the pipeline is trained.

🤖 Model
Logistic Regression without PCA
Features: 64
Accuracy: 97.22%
PCA + Logistic Regression
Features: 31
Variance retained: 90.06%
Accuracy: 94.72%
📈 Model Comparison
Model	Features	Accuracy
Logistic Regression	64	97.22%
PCA + Logistic Regression	31	94.72%

PCA reduced the number of features by approximately 52%, but classification accuracy decreased by about 2.5 percentage points.

🔍 Classification Performance

The PCA + Logistic Regression model achieved:

Accuracy: 94.72%
Macro F1: 0.95
Weighted F1: 0.95

The strongest-performing classes included digits 2, 3, and 5, while digits 1 and 8 were comparatively more difficult to classify.

The confusion matrix showed that the model particularly confused some handwritten 8s with 1s.

📊 Visualizations

The project includes:

Digit image visualization
PCA explained variance plot
2D PCA visualization
Confusion matrix
Model accuracy comparison
🛠️ Technologies Used
Python
NumPy
Pandas
Matplotlib
Scikit-learn
StandardScaler
PCA
Logistic Regression
📂 Project Structure
handwritten-digit-pca/
│
├── notebook/
│   └── notebook.ipynb
│
├── images/
│   ├── Digits_Visualized_Using_PCA.png
│   ├── pca_explained_variance.png
│   ├── Confusion_Matrix_PCA_with_Logistic Regression.png
│   └── model_comparision.png
├
└── README.md
🚀 How to Run

Clone the repository:

git clone https://github.com/yourusername/handwritten-digit-pca.git

Install the dependencies:

pip install -r requirements.txt

Launch Jupyter Notebook:

jupyter notebook

Open:

notebook/handwritten_digit_pca.ipynb
📦 Requirements
numpy
pandas
matplotlib
scikit-learn
jupyter
💡 Key Learning

This project demonstrates an important machine learning concept:

Dimensionality reduction does not always improve model accuracy.

In this experiment, PCA reduced the input from 64 to 31 features while retaining approximately 90% of the variance, but the original 64-feature Logistic Regression model achieved higher accuracy.

Therefore, PCA can be valuable when the goal is dimensionality reduction, visualization, or computational efficiency, even when it does not produce the highest classification accuracy.

👨‍💻 Author

Jaskaran Singh

Built as part of practical machine learning and unsupervised learning practice.
