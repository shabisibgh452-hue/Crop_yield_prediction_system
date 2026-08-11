# 🌾 Crop Yield Prediction System

A Machine Learning based web application that predicts crop yield using agricultural and environmental factors. The application is built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 📌 Features

- Predict Crop Yield using Machine Learning
- User-friendly Streamlit interface
- Dark Theme Dashboard
- Real-time prediction
- Professional and responsive UI

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Plotly

---

## 📂 Project Structure

```
Crop_Yield_Prediction/
│── dataset/
│   ├── yield_df.csv
│   └── crop_cleaned.csv
│
│── models/
│   ├── best_crop_model.pkl
│   ├── area_encoder.pkl
│   └── item_encoder.pkl
│
│── app.py
│── config.py
│── training.py
│── preprocessing.py
│── requirements.txt
│── README.md
```

---

## 📊 Input Features

- Area
- Crop (Item)
- Year
- Average Rainfall (mm/year)
- Pesticides (Tonnes)
- Average Temperature

---

## 🎯 Output

The model predicts the **Crop Yield (hg/ha)** based on the provided input values.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/crop-yield-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🤖 Machine Learning Model

- Random Forest Regressor
- Trained using historical crop production data
- Encoded categorical features using LabelEncoder