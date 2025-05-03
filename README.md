# 💻 Computer Hardware Performance Analysis

This project is part of a university course on Digital Data Processing (Traitement Numérique des Données). It focuses on predicting the Published Relative Performance (PRP) of computer systems based on their hardware specifications using real-world data.

## 📁 Dataset

- **Source:** [UCI Machine Learning Repository – Computer Hardware](https://archive.ics.uci.edu/ml/datasets/Computer+Hardware)
- **Instances:** 209 computers
- **Features used:** memory (MMIN, MMAX), cache size (CACH), number of channels (CHMIN, CHMAX), cycle time (MYCT)
- **Target variable:** PRP (Published Relative Performance)

## 🧪 Objectives

- Load and clean the dataset
- Normalize the numerical features
- Apply and compare multiple regression models (Linear Regression and Random Forest)
- Visualize prediction results and feature importance
- Perform Principal Component Analysis (PCA) to reduce dimensionality and explore data structure

## 🧰 Tools & Technologies

- **Python 3**
- **Pandas** – data manipulation
- **Scikit-learn** – modeling, evaluation, PCA
- **Matplotlib & Seaborn** – data visualization
- **Jupyter Notebook / VS Code** – development environment
- **LaTeX (Overleaf)** – professional report writing

## 📊 Models Used

- **Multiple Linear Regression** — simple and interpretable baseline model to predict PRP from hardware features.
- **Random Forest Regressor** — non-linear model that improved prediction accuracy and provided insights on feature importance.
- **Principal Component Analysis (PCA)** — dimensionality reduction technique used to explore data structure and visualize relationships between observations.
- **K-means Clustering** — unsupervised method used to identify groups of similar computer configurations based on hardware.
- **Univariate Statistical Analysis** — boxplots, histograms, and descriptive statistics to detect outliers, skewness, and justify data normalization.
- **Correlation Matrix** — to evaluate feature relationships and check for multicollinearity or redundancy.


## 📈 Key Results

- Random Forest achieved better predictive performance than Linear Regression (lower RMSE and higher R²)
- PCA revealed that the first two components capture approximately 70% of the variance
- Strong correlation observed between memory-related features and PRP
- MYCT and CHMAX showed large variance and outliers
- The most influential hardware features were MMAX (max memory), CACH (cache size), and CHMIN (min channels)

## 📎 License

This project is intended for academic and educational use only.



