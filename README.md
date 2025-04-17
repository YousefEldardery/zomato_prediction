# 🍽️ Zomato Restaurant Rating Prediction

This project is a machine learning model that predicts Zomato restaurant ratings based on various features like online ordering availability, table booking options, votes, location, restaurant type, and cuisine. The model is deployed with a Streamlit app that provides a simple user interface for making predictions.

## 📌 Project Goals

- Analyze and preprocess the Zomato dataset.
- Build a regression model to predict restaurant ratings.
- Deploy the model in a Streamlit web application.

## 📁 Files in the Repository

| File | Description |
|------|-------------|
| `zomato-yousef-eldardery.ipynb` | Jupyter Notebook containing EDA, data preprocessing, model training, and evaluation |
| `deployment zomato.py` | Python script for deploying the model with Streamlit |
| `model.pkl` *(optional)* | Saved trained model (should be added if available) |
| `README.md` | Project documentation |

## 🔍 Features Used

- **online_order**: Whether the restaurant accepts online orders.
- **book_table**: Whether the restaurant allows table bookings.
- **votes**: Number of customer votes.
- **location**: The restaurant's location.
- **rest_type**: Type of restaurant (e.g., cafe, fine dining).
- **cuisines**: The cuisines served.

## 🔧 Technologies Used

- Python
- Pandas, NumPy, Scikit-learn
- Streamlit (for deployment)
- Matplotlib, Seaborn (for visualization)

## 🧠 Model Training Steps

- Cleaned and encoded categorical features.
- Handled missing values.
- Used regression models (e.g., Linear Regression, Random Forest).
- Evaluated performance using metrics like **R² score**.

## 📊 Sample Output

The app takes user inputs for restaurant details and outputs the **predicted rating** using the trained model.

## 📌 To Do

- Improve UI with Streamlit widgets.
- Add feature importance visualization.
- Add option to upload a CSV and predict in batch.

## 🙌 Author

**Yousef Eldardery**  
Connect on [GitHub](https://github.com/YousefEldardery)


