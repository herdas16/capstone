from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Ganti informasi koneksi database sesuai dengan konfigurasi Google Cloud SQL Anda
db_user = 'root'
db_password = 'Financify123#'
db_name = 'financify'
cloud_sql_connection_name = 'capstone-project-406514:us-west4:financify'

# Ganti URI koneksi MySQL dengan URI koneksi MySQL untuk Google Cloud SQL
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+mysqlconnector://{db_user}:{db_password}@/{db_name}?unix_socket=/cloudsql/{cloud_sql_connection_name}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Load the pre-trained Keras model
model = load_model('./Final_Model/all_cities_lstm_model_v3.h5')

# Initialize scaler
scaler = MinMaxScaler()

# Load data
file_path = './Dataset/inflasi_v3.csv'
df = pd.read_csv(file_path)

# Select relevant columns
df = df[['City', 'Month', 'Year', 'Inflation', 'CPI']]

# Remove 'KOTA' from the 'City' column
df['City'] = df['City'].str.replace('KOTA ', '', regex=False)

# Check if 'Inflation' and 'CPI' columns already contain numeric values
for col in ['Inflation', 'CPI']:
    if not pd.api.types.is_numeric_dtype(df[col]):
        # Replace commas with dots and convert to float
        df[col] = df[col].str.replace(',', '.').astype(float)

# Sort the data
df = df.sort_values(by=['City', 'Year', 'Month'])

# Scale inflation data and other numerical features
scaler.fit_transform(df[['Inflation', 'CPI']].values)

# Separate data into time series for each city
city_data = {}
for city in df['City'].unique():
    city_data[city] = df[df['City'] == city][['Inflation', 'CPI']].values

# Combine data for all cities into a single time series
all_cities_data = np.concatenate(list(city_data.values()))

# Function to create time series sequences
def create_time_series(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps + 1):
        a = data[i:(i + time_steps), :]
        X.append(a)
        y.append(data[i + time_steps - 1, 0])  # Assuming 'Inflation' is in the first column
    return np.array(X), np.array(y)

# Hyperparameters
time_steps = 12
n_features = 2

@app.route('/predict', methods=['POST'])
def predict():
    error = None
    prediction = None
    trend = None
    time_required = None
    time_required_predicted = None

    try:
        city = request.form['city']
        goal = float(request.form['goal'])
        income = float(request.form['income'])
        expenses = float(request.form['expenses'])
        historical_inflation = city_data[city]

        # Simpan data ke MySQL
        save_to_mysql(city, goal, income, expenses)

        input_data = scaler.transform(historical_inflation[-time_steps:])
        input_data = input_data.reshape((1, time_steps, n_features))

        # Make prediction
        predicted_inflation = model.predict(input_data)

        # Inverse transform the prediction to get the actual value
        predicted_inflation_actual = scaler.inverse_transform(np.concatenate([predicted_inflation, np.zeros_like(predicted_inflation)], axis=1))
        predicted_inflation_actual = predicted_inflation_actual[:, 0]
        prediction = predicted_inflation_actual[0]

        # Compare with the most recent actual inflation value
        last_actual_inflation = historical_inflation[-1, 0]  # Assuming 'Inflation' is in the first column

        # Determine the trend
        if prediction > last_actual_inflation:
            trend = 'up'
        elif prediction < last_actual_inflation:
            trend = 'down'
        else:
            trend = 'unchanged'

        # Calculate time required based on predicted inflation
        years_to_goal, remaining_months = calculate_time_to_goal(goal, income, expenses, prediction)

        prediction = round(prediction * 100, 2)

        # Simpan data ke MySQL
        save_result(prediction, trend, years_to_goal, remaining_months)

    except Exception as e:
        error = str(e)

    return jsonify({
        "prediction": prediction,
        "trend": trend,
        "time_required": {
            "years_to_goal": years_to_goal,
            "remaining_months": remaining_months
        }
    })

def calculate_time_to_goal(goal, income, expenses, savings):
    # Calculate monthly savings
    monthly_savings = income - expenses

    # Calculate the number of months required to reach the goal
    months_to_goal = int((goal - savings) / monthly_savings)

    # Convert months to years and months
    years_to_goal = months_to_goal // 12
    remaining_months = months_to_goal % 12

    return years_to_goal, remaining_months

def save_to_mysql(city, goal, income, expenses):
    try:
        # Membuka kursor
        cursor = db.session.connection().cursor()

        # Menjalankan query INSERT
        cursor.execute('INSERT INTO predict (city, goal, income, expenses) VALUES (%s, %s, %s, %s)',
                       (city, goal, income, expenses))

        # Melakukan commit untuk menyimpan perubahan pada database
        db.session.commit()

        # Menutup kursor
        cursor.close()

        return True  # Berhasil menyimpan

    except Exception as e:
        # Mengembalikan False jika ada kesalahan
        print(f"Error: {e}")
        db.session.rollback()  # Rollback perubahan jika terjadi kesalahan
        return False

def save_result(prediction, trend, years_to_goal, remaining_months):
    try:
        # Membuka kursor
        cursor = db.session.connection().cursor()

        # Menjalankan query INSERT
        cursor.execute('INSERT INTO result (prediction, trend, years_to_goal, remaining_months) VALUES (%s, %s, %s, %s)',
                       (prediction, trend, years_to_goal, remaining_months))

        # Melakukan commit untuk menyimpan perubahan pada database
        db.session.commit()

        # Menutup kursor
        cursor.close()

        return True  # Berhasil menyimpan

    except Exception as e:
        # Mengembalikan False jika ada kesalahan
        print(f"Error: {e}")
        db.session.rollback()  # Rollback perubahan jika terjadi kesalahan
        return False

if __name__ == '__main__':
    app.run(debug=True)
