Breast Cancer Classification Project
About
This project uses machine learning to predict whether a breast tumor is benign or malignant based on medical measurements from the Breast Cancer Wisconsin dataset.

Models Used
Logistic Regression
Decision Tree
KNN
Naive Bayes
SVM

What I Did
Loaded and explored the dataset
Split data into training and testing sets
Applied feature scaling where needed
Trained and evaluated multiple models
Compared their performance
Built a Streamlit app for predictions
Run the Project

Install dependencies:
pip install -r requirements.txt

Train models:
python train.py

Start the app:
streamlit run app.py

Project Files
train.py – model training
app.py – Streamlit application
test_data.csv – sample input data
model/ – saved models and results

Result
Among all models tested, SVM and Logistic Regression gave the best performance on this dataset.

Author
Ritu Sahu