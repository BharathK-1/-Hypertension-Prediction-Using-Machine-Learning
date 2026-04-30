📌 Project Overview

Hypertension (High Blood Pressure) is a leading cause of cardiovascular diseases worldwide.
Machine Learning can help identify at-risk individuals by learning from patient data.

This project includes:

Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Model Training & Evaluation
Saving the best ML model
Prediction script or web app (if included in your project)
📁 Project Structure

Here is the typical structure from your uploaded ZIP (adjusted based on standard ML project templates):

📦 Hypertension Prediction
├── dataset/                     # CSV file with medical data
├── notebooks/                   # Jupyter notebook (EDA & model training)
├── src/
│   ├── preprocessing.py         # Data cleaning & transformation
│   ├── train_model.py           # Model training code
│   ├── model.pkl                # Saved machine learning model
│   ├── predict.py               # Prediction pipeline
├── app.py                       # Flask/Streamlit app (if provided)
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
🧠 Machine Learning Models Tested

Your project typically includes the following algorithms:

Logistic Regression
Random Forest Classifier
Decision Tree Classifier
Support Vector Machine (SVM)
XGBoost (optional)

The best accuracy model is exported as model.pkl.

🔧 Technologies Used
Python 3.x
Pandas
NumPy
Scikit-Learn
Matplotlib / Seaborn
Flask / Streamlit (if applicable)
🚀 How to Run the Project
1️⃣ Install Libraries
pip install -r requirements.txt
2️⃣ Train the Model
python src/train_model.py
3️⃣ Make Predictions
python src/predict.py
4️⃣ Run the Web App (if available)
python app.py
📊 Model Evaluation

Performance metrics included:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC–AUC Curve

These results help determine the best model for hypertension risk prediction.

📈 Future Improvements
Add deep learning models
Deploy using Flask, FastAPI, or Streamlit
Add real-time prediction dashboard
Improve feature engineering
