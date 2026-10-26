# Credit-Card-Risk-Assessment-
Machine Learning | Python | Flask

#Deployment-https://credit-card-risk-assessment-685i.onrender.com

A machine-learning web application that predicts whether a customer's credit profile is classified as Good or Bad using customer and credit-related information.

</div>

📌 About the Project

CredWise is a credit repayment risk prediction project built with Python and machine learning. It uses a trained Gradient Boosting Classifier to classify a customer's profile. A Flask web application collects user inputs, applies the required preprocessing and scaling, and displays the prediction with a probability when available.

Note: This project is intended for learning and demonstration. It is not a substitute for professional financial assessment and should not be used as the sole basis for lending or credit decisions.

✨ Features

Predicts a Good or Bad credit profile.

Accepts numerical and categorical customer information.

Applies numerical transformations, categorical encoding, and feature scaling.

Uses a saved machine-learning model and scaler.

Displays a prediction result and model probability when available.

Includes a Python data-preprocessing and model-training workflow.

Provides a Flask web interface.

🧰 Technologies Used

Technology

Purpose

Python

Core programming language

Pandas, NumPy

Data handling and numerical operations

scikit-learn

Machine learning, encoding, scaling, and evaluation

imbalanced-learn (SMOTE)

Training-data class balancing

SciPy

Yeo–Johnson transformation

Flask

Web application

Pickle

Saving and loading the trained model and scaler

Gunicorn

WSGI server for deployment

🔄 Project Workflow

Customer Dataset
       ↓
Data Cleaning and Missing-Value Handling
       ↓
Outlier Treatment and Feature Selection
       ↓
Categorical Feature Encoding
       ↓
SMOTE Class Balancing (training data)
       ↓
StandardScaler
       ↓
Gradient Boosting Classifier
       ↓
Model Evaluation and Saved Model
       ↓
Flask Web Application
       ↓
Good / Bad Credit Profile Prediction

📥 Input Features

The web form collects the following information:

Age

Number of open credit lines and loans

Number of real-estate loans or lines

Monthly income

Gender

Region

Housing status

Occupation

Education

The application transforms the inputs into the feature format expected by the saved model and scaler before making a prediction.

🗂️ Project Structure

Keep your repository organized like this (update names if your files differ):

CredWise/
├── app.py                       # Flask web application
├── main.py                      # Data preprocessing and model training
├── templates/
│   └── index.html                # Frontend page
├── creditcard.csv                # Dataset (do not publish sensitive data)
├── creditcard_model.pkl          # Trained Gradient Boosting model
├── scaled model.pkl               # Fitted StandardScaler
├── requirements.txt              # Python dependencies
├── Procfile                      # Deployment start command
└── README.md                      # Project documentation

Important: Flask's render_template("index.html") normally expects index.html inside the templates/ folder. The model and scaler filenames must exactly match the paths in app.py.

⚙️ Installation and Setup

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CredWise

Replace <YOUR_GITHUB_REPOSITORY_URL> with your repository URL.

2. Create a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

Make sure the requirements file includes all packages imported by your code, including Flask, NumPy, SciPy, scikit-learn, Pandas, and imbalanced-learn where applicable.

4. Confirm the model files

Ensure these files exist at the paths configured in app.py:

creditcard_model.pkl
scaled model.pkl

If your files have different names, rename them or update MODEL_PATH and SCALER_PATH in the Flask application.

5. Run the application

python app.py

Open the local URL shown in the terminal. It is usually:

http://127.0.0.1:5000

🚀 Deployment

For a Python hosting service that supports Gunicorn:

Push the project to your GitHub repository.

Install dependencies using pip install -r requirements.txt.

Set the start command to:

gunicorn app:app

Confirm the Python module name matches the filename containing the Flask variable app.

Confirm the model and scaler files are available in the deployed environment.

Test the deployed page with valid and invalid inputs.

📊 Model Evaluation

The training script evaluates the model using:

Accuracy

Confusion matrix

Precision

Recall

F1-score

Add your actual evaluation results here after running the training script. No accuracy score is listed because a verified result was not provided.

🧪 Testing Checklist

The home page loads.

The form accepts valid inputs.

A Good/Bad prediction is displayed.

Probability is shown when available.

Invalid values return a clear error.

The model and scaler load successfully.

Feature order and category mappings match training.

The deployed application runs successfully.

🔒 Responsible Use

Do not upload private customer data or personal financial information to a public repository.

Only load trusted Pickle files; unpickling untrusted files can execute malicious code.

Treat predictions as estimates, not guarantees.

Evaluate accuracy, fairness, and limitations before considering any real-world use.

Do not use this demo as the sole basis for credit approval or rejection.

🔮 Future Enhancements

Add a dashboard for prediction and model-performance summaries.

Add explainability with feature importance or SHAP.

Improve input validation and automated testing.

Track model versions and monitor prediction quality.

Evaluate fairness and probability calibration.

👨‍💻 Author

Sagar Datti

Data Science | Machine Learning | AI Engineering
