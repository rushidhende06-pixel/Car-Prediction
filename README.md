Project overview
The application lets a user enter details such as manufacturer, model,
production year, fuel type, engine size, mileage, cylinders, airbags,
and other car attributes. It then prepares the input data and predicts
an estimated car price.
- Problem type: Supervised Machine Learning -- Regression
- Target variable: Price
- Model: RandomForestRegressor
- Application: Streamlit
- Dataset: Used Car Price Prediction dataset
Features
- Interactive car-details form
- Estimated price displayed after submitting the form
- Data cleaning and feature preparation
- Missing-value handling
- One-hot encoding for categorical features
- Scaling for numeric features
- Model training cached by Streamlit to avoid retraining on every
  interaction
Dataset
The app loads the CSV dataset from:

car_price_prediction.csv

The target column is Price. The app derives the Engine, Turbo, and
Age features from the original columns.

Note: The application needs an internet connection to download the
dataset when it starts for the first time.
Project structure
Car-Price-Prediction/
├── app.py
├── requirements.txt
└── README.md
Run locally
1. Clone the repository
git clone https://github.com/rushidhende06-pixel/Car-Price-Prediction.git
cd Car-Price-Prediction
If your repository has a different name, use its actual GitHub URL and
folder name.
2. Create and activate a virtual environment
Windows:
python -m venv venv
venv\Scripts\activate
macOS / Linux:
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Start the Streamlit app
streamlit run app.py
The app will open in your browser. If it does not, open the local URL
shown in the terminal, usually http://localhost:8501.
Deploy on Streamlit Community Cloud
1. Push app.py, requirements.txt, and README.md to your GitHub
   repository.
2. Open Streamlit Community Cloud.
3. Sign in with GitHub and choose Create app.
4. Select your repository and the main branch.
5. Set the main file path to app.py.
6. Choose Python 3.12 in Advanced settings.
7. Deploy the app.
Important notes
- The app trains a Random Forest model from the dataset when it first
  loads. Streamlit caches the trained model for subsequent reruns.
- The predicted amount uses the same currency/unit as the dataset's
  Price column. Confirm the dataset's currency before presenting
  predictions as a market price.
- A prediction is an estimate, not a guaranteed resale or market
  value.
- The model and preprocessing steps in this app are trained together
  in one scikit-learn pipeline to keep inference preprocessing
  consistent with training.
- The app's training approach is a Random Forest implementation. Do
  not describe it as the Linear Regression model from a separate
  notebook unless the app is updated to load and use that exact
  trained model.
Technologies used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regression
Author
Rushikesh Dhende
