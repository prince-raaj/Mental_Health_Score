# 🧠 Student Mental Health Score Prediction

A machine learning regression project that predicts a student's **Mental Health Score** using academic, lifestyle, social-media usage, and demographic factors.

> **Note:** This project is intended for educational and analytical purposes. The prediction should not be treated as a clinical diagnosis or medical assessment.

---

## 📌 Project Overview

Students' mental well-being can be influenced by multiple lifestyle and behavioural factors such as social-media usage, sleep, physical activity, study hours, and stress levels.

The objective of this project is to build a **supervised machine learning regression model** that predicts the `Mental_Health_Score` from these features.

The project covers the complete machine learning workflow:

* Data loading and inspection
* Data cleaning
* Exploratory Data Analysis
* Feature engineering
* Outlier analysis
* Skewness analysis
* Feature preprocessing
* Model training
* Model comparison
* Hyperparameter tuning
* Cross-validation
* Model evaluation

---

## 🎯 Problem Statement

**Goal:** Predict a student's `Mental_Health_Score` based on available demographic, academic, lifestyle, and social-media-related information.

### Machine Learning Type

* **Learning:** Supervised Learning
* **Problem:** Regression
* **Target:** `Mental_Health_Score`

---

## 📊 Dataset

The dataset contains **5,000 records initially** and **13 columns**. After identifying and removing 2 duplicate records, the working dataset contains **4,998 records**.

### Features

| Feature                   | Description                        |
| ------------------------- | ---------------------------------- |
| `Age`                     | Student age                        |
| `Gender`                  | Student gender                     |
| `Country`                 | Student's country                  |
| `Academic_Level`          | Academic level                     |
| `Most_Used_Platform`      | Most-used social media platform    |
| `Purpose_Of_Use`          | Main purpose of social media usage |
| `Avg_Daily_Usage_Hours`   | Average daily social-media usage   |
| `Daily_Unlocks`           | Number of daily device unlocks     |
| `Study_Hours`             | Daily study hours                  |
| `Physical_Activity_Hours` | Physical activity hours            |
| `Sleep_Hours_Per_Night`   | Average sleep hours                |
| `Stress_Level`            | Stress level                       |
| `Mental_Health_Score`     | Target variable                    |

---


## 🧰 Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## 📁 Project Structure

```text
student-mental-health-score-prediction/
│
├── Mental_Health_Score.ipynb
│
├── dataset/
│   └── Student Social Media And Mental Health Impact.csv
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/akashgoswami139/student-mental-health-score-prediction.git
```



## 🔮 Future Improvements

Planned improvements for the project include:

* Feature importance analysis
* Residual/error analysis
* Further model evaluation
* Final model selection
* Saving the trained preprocessing + model pipeline using Joblib
* Building an interactive Streamlit application
* Deploying the Streamlit application

The Streamlit application will be maintained separately from this machine-learning repository.

---

## 📝 Key Learning Outcomes

Through this project, I practised:

* Exploratory Data Analysis
* Data cleaning
* Feature engineering
* Handling numerical and categorical features
* Ordinal and one-hot encoding
* Feature scaling
* Transformation of skewed features
* Scikit-learn pipelines
* Column Transformer
* Regression modelling
* Random Forest
* Cross-validation
* Hyperparameter tuning
* Model evaluation and comparison

---

## ⚠️ Disclaimer

This project is an educational machine learning project. A predicted mental health score is **not a medical diagnosis**, and the model should not be used as a substitute for professional mental-health assessment or treatment.

---

## 👨‍💻 Author

**Akash Goswami**
