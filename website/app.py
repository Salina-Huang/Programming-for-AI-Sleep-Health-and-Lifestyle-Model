from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global variables for storing the model and scaler
model = None
scaler = None
feature_columns = None
numerical_columns = None


def train_model():
    """Train the sleep quality prediction model"""
    global model, scaler, feature_columns
    
    # Load the original dataset (not the already processed version)
    data = pd.read_csv('https://raw.githubusercontent.com/Salina-Huang/Programming-for-AI-Sleep-Health-and-Lifestyle-Model/refs/heads/main/Sleep_health_and_lifestyle_dataset.csv')
    
    # Display data information
    print(f"Original dataset shape: {data.shape}")
    print(f"Original feature columns: {data.columns.tolist()}")
    
    # Prepare features and target variable
    target_variable = "Quality of Sleep"
    
    # Select only the required feature columns
    selected_features = [
        'Age', 'Sleep Duration', 'Physical Activity Level', 
        'Stress Level', 'Heart Rate', 'Daily Steps', 
        'Occupation', 'BMI Category', 'Sleep Disorder'
    ]
    
    X = data[selected_features]
    y = data[target_variable]
    
    # Perform one-hot encoding for categorical features
    categorical_columns = ['Occupation', 'BMI Category', 'Sleep Disorder']
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=False)
    
    # Save feature column names
    feature_columns = X_encoded.columns.tolist()
    
    # Standardize numerical features
    numerical_columns = ['Age', 'Sleep Duration', 'Physical Activity Level', 
                        'Stress Level', 'Heart Rate', 'Daily Steps']
    
    scaler = StandardScaler()
    X_encoded[numerical_columns] = scaler.fit_transform(X_encoded[numerical_columns])
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Train the random forest model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save the model and scaler
    with open('sleep_quality_model.pkl', 'wb') as f:
        pickle.dump((model, scaler, feature_columns, numerical_columns), f)
    
    print("Model trained and saved successfully!")
    print(f"Number of feature columns: {len(feature_columns)}")
    print(f"Numerical features: {numerical_columns}")


def load_model():
    """Load the saved model and scaler"""
    global model, scaler, feature_columns, numerical_columns
    try:
        with open('sleep_quality_model.pkl', 'rb') as f:
            model, scaler, feature_columns, numerical_columns = pickle.load(f)
        print("Model loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model file not found. Training new model...")
        train_model()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to train new model...")
        train_model()
        return True


@app.route('/api/predict', methods=['POST'])
def predict_sleep_quality():
    """Sleep quality prediction API endpoint"""
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            print("Model not loaded, calling load_model()...")
            if not load_model():
                print("Failed to load or train model")
                return jsonify({"error": "Failed to load or train model"}), 500
        
        # Get request data
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Create input data dictionary (Note: frontend sends original field names, consistent with dataset)
        input_data = {
            'Age': data.get('Age', 0),
            'Sleep Duration': data.get('Sleep Duration', 0),
            'Physical Activity Level': data.get('Physical Activity Level', 0),
            'Stress Level': data.get('Stress Level', 0),
            'Heart Rate': data.get('Heart Rate', 0),
            'Daily Steps': data.get('Daily Steps', 0),
        }
        
        # Add occupation features (one-hot encoding)
        occupation = data.get('Occupation', '')
        print(f"Processing occupation: {occupation}")
        for col in feature_columns:
            if col.startswith('Occupation_'):
                input_data[col] = 1 if col == f'Occupation_{occupation}' else 0
        
        # Add BMI category features (one-hot encoding)
        bmi_category = data.get('BMI Category', '')
        print(f"Processing BMI Category: {bmi_category}")
        for col in feature_columns:
            if col.startswith('BMI Category_'):
                input_data[col] = 1 if col == f'BMI Category_{bmi_category}' else 0
        
        # Add sleep disorder features (one-hot encoding)
        sleep_disorder = data.get('Sleep Disorder', '')
        print(f"Processing Sleep Disorder: {sleep_disorder}")
        for col in feature_columns:
            if col.startswith('Sleep Disorder_'):
                input_data[col] = 1 if col == f'Sleep Disorder_{sleep_disorder}' else 0
        
        # Create feature matrix
        print(f"Feature columns: {feature_columns}")
        print(f"Input data: {input_data}")
        X_input = pd.DataFrame([input_data])[feature_columns]
        print(f"X_input shape: {X_input.shape}")
        
        # Only standardize numerical features
        print(f"Numerical columns: {numerical_columns}")
        X_input[numerical_columns] = scaler.transform(X_input[numerical_columns])
        
        # Predict sleep quality
        prediction = model.predict(X_input)
        print(f"Prediction: {prediction[0]}")
        
        # Return prediction result
        return jsonify({
            "prediction": float(prediction[0]),
            "message": "Prediction successful"
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/features', methods=['GET'])
def get_features():
    """Get all feature information"""
    try:
        if feature_columns is None:
            if not load_model():
                return jsonify({"error": "Failed to load or train model"}), 500
        
        return jsonify({
            "features": feature_columns,
            "message": "Features retrieved successfully"
        })
        
    except Exception as e:
        print(f"Error getting features: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 加载或训练模型
    load_model()
    # 启动Flask服务器
    app.run(debug=True)