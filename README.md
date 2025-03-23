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

- Multiple Linear Regression
- Random Forest Regressor
- Principal Component Analysis (PCA)


## 📈 Key Results

- Random Forest achieved better predictive performance than Linear Regression (lower RMSE and higher R²)
- PCA revealed that the first two components capture approximately 70% of the variance
- The most influential hardware features were MMAX (max memory), CACH (cache size), and CHMIN (min channels)




