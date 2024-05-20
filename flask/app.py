from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('disease.csv')
# Add a separate disease ID column with unique IDs for each disease
unique_diseases = dataset['Disease'].unique()
disease_id_mapping = {disease: i+1 for i, disease in enumerate(unique_diseases)}
dataset['Disease_ID'] = dataset['Disease'].map(disease_id_mapping)

# Preprocess the data
# Convert categorical variables into numerical values
label_encoder = LabelEncoder()
dataset['Outcome Variable'] = dataset['Outcome Variable'].map({'Positive': 1, 'Negative': 0})
for column in dataset.columns[2:]:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Separate features and target variable
X = dataset.drop(columns=['Disease', 'Outcome Variable', 'Disease_ID'])  # Exclude Outcome Variable and Disease_ID
y = dataset['Disease_ID']

# Convert 'Yes'/'No' values to 1/0
X[X == 'Yes'] = 1
X[X == 'No'] = 0

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train the Gradient Boosting classifier
gbm_classifier = GradientBoostingClassifier()
gbm_classifier.fit(X, y)

# Predict disease using GBM
def predict_disease(symptoms):
    symptoms = [1 if value.lower() == 'yes' else 0 for value in symptoms.values()]
    predicted_disease_id = gbm_classifier.predict([symptoms])[0]
    predicted_disease = dataset.loc[dataset['Disease_ID'] == predicted_disease_id, 'Disease'].iloc[0]
    return predicted_disease

# Predict diseases for the entire dataset
y_pred = gbm_classifier.predict(X)

# Calculate Mean Squared Error
mse = mean_squared_error(y, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = {
        'Fever': request.form['Fever'],
        'Cough': request.form['Cough'],
        'Fatigue': request.form['Fatigue'],
        'Difficulty Breathing': request.form['Difficulty Breathing'],
        'Age': request.form['Age'],
        'Gender': request.form['Gender'],
        'Blood Pressure': request.form['Blood Pressure'],
        'Cholesterol Level': request.form['Cholesterol Level']
    }
    predicted_disease = predict_disease(symptoms)
    return render_template('result.html', disease=predicted_disease, mse=mse, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
