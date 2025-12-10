# Sleep Quality Prediction System User Guide

This is a machine learning-based sleep quality prediction system. Users can input their personal health information, and the system will predict their sleep quality and provide improvement suggestions.

## Project Overview

- **Features**: Predicts sleep quality (1-10 scale) based on user input health data and provides personalized improvement suggestions
- **Technology Stack**:
  - Backend: Python + Flask
  - Machine Learning: scikit-learn (Random Forest Regression)
  - Frontend: HTML + CSS + JavaScript

## Project Structure

```
website/
├── app.py              # Backend server and machine learning model
├── sleep_quality_predictor.html  # Frontend user interface
├── sleep_quality_model.pkl  # Pre-trained machine learning model
├── check_model.py      # Model checking script
├── README.md           # This documentation file
└── venv/               # Python virtual environment (optional)
```

## Running Steps

### 1. Environment Setup

#### 1.1 Python Environment Requirements
- Python 3.6 or higher
- pip package manager

#### 1.2 Install Dependencies

Open a command line terminal, navigate to the project directory (`f:\桌面\website`), and execute the following command:

```bash
# Install required Python libraries
pip install flask flask-cors pandas numpy scikit-learn
```

### 2. Start the Backend Server

**Note**: You must start the backend server first before the frontend HTML page can work properly!

Execute the following command in the project directory:

```bash
python app.py
```

After successful startup, the command line will display information similar to:

```
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
* Debugger is active!
```

### 3. Open the Frontend Page

Open the `sleep_quality_predictor.html` file in a browser:

1. Locate the `sleep_quality_predictor.html` file in the project directory
2. Double-click the file to open it directly in the browser
3. Or right-click and select "Open with", then choose any modern browser

### 4. Use the System

1. **Fill in the Form**: On the opened page, fill in the following information:
   - Age
   - Sleep Duration (hours)
   - Physical Activity Level (minutes per day)
   - Stress Level (1-10 scale)
   - Heart Rate (beats per minute)
   - Daily Steps
   - Occupation
   - BMI Category
   - Sleep Disorder (if any)

2. **Submit Prediction**: Click the "Predict Sleep Quality" button

3. **View Results**:
   - The page will display the predicted sleep quality score (1-10 scale)
   - Personalized sleep improvement suggestions will be provided

## Frequently Asked Questions

### Q: Can I run the HTML file directly without starting the backend server?
A: **No**. This system uses a frontend-backend separation architecture:
- The HTML file is the frontend interface responsible for user input and result display
- The Python backend server provides prediction functionality and machine learning model
- The frontend calls the backend service via API to obtain prediction results

If you don't start the backend server, the HTML page can open, but clicking the prediction button will display an error message.

### Q: What should I do if the server fails to start?
A: Check the following points:
1. Is Python correctly installed?
2. Are all dependency packages installed successfully?
3. Is port 5000 occupied by another program? (You can try changing the port)

### Q: What if the prediction results are inaccurate?
A: The system uses a model trained on a public dataset and may have certain errors. You can:
1. Check if the input data is accurate
2. Try providing more diverse input combinations

### Q: How to modify the backend port?
A: Find the following code in the `app.py` file:
```python
app.run(debug=True)
```
Modify it to:
```python
app.run(debug=True, port=8080)  # Use port 8080
```

## Technical Notes

### Model Training
- When running `app.py` for the first time, the system will automatically download the dataset and train the model
- The trained model will be saved in the `sleep_quality_model.pkl` file
- Subsequent runs will directly load the trained model

### API Endpoints
- `POST /api/predict`: Receives user data and returns prediction results
- `GET /api/features`: Gets the list of features supported by the system

## Stop the Server

In the command line terminal, press the `Ctrl + C` key combination to stop the backend server.

## Notes

1. Ensure network connectivity is normal (required for downloading the dataset during first run)
2. Do not start multiple backend server instances simultaneously
3. Do not use debug mode in production environments
4. If you modify the `app.py` file, the server will automatically restart (in debug mode)

---

Enjoy using it! If you have any questions, please check the system configuration or contact the project developer.