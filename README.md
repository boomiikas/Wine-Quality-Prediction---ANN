# Wine Quality Prediction Web App 🍷

This is a **Gradio web application** for predicting **wine quality** using an **Artificial Neural Network (ANN)** model trained on physicochemical features.

**DEMO LINK :** https://huggingface.co/spaces/boomiikas/Wine-Quality-Prediction

## Features
- Predicts wine quality as **Low**, **Medium**, or **High**.
- Uses 11 physicochemical features:
  - Fixed Acidity
  - Volatile Acidity
  - Citric Acid
  - Residual Sugar
  - Chlorides
  - Free Sulfur Dioxide
  - Total Sulfur Dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- Interactive sliders for each feature.
- Clean sidebar-style layout grouped by feature types.

## Model Improvements & Observations
- Attempted to improve accuracy using **SMOTE** and **class weights**.
  - SMOTE accuracy: 0.545
  - Class weights accuracy: 0.625
  - Weighted F1 Score: 0.611 → Considering class imbalance, the model is moderately performing.

### Detailed Class Performance
- **High**: 31/37 correct → good recall, 6 misclassified as Low/Medium.
- **Low**: 102/128 correct → strong performance.
- **Medium**: Only 37/107 correct → struggles the most; many Medium samples predicted as High (36) or Low (34).

**Observation**: The model tends to overpredict High for Medium and sometimes Low. Medium is the class that suffers the most.

## Requirements
- Python 3.8+
- TensorFlow
- Gradio
- NumPy

Install required packages:
```bash
pip install tensorflow gradio numpy
```

## Files
- `app.py`: Main Gradio application.
- `model.h5`: Trained ANN model.

## Usage
1. Place `model.h5` in the same directory as `app.py`.
2. Run the Gradio app:
```bash
python app.py
```
3. A browser window will open with sliders to input wine features.
4. Click **Predict Quality** to see the predicted wine quality.

## Notes
- The app uses an **Artificial Neural Network (ANN)** trained on physicochemical wine features.
- No scaling or preprocessing is required for inputs.
- The quality mapping in the ANN output is as follows:
  - `0`: Low
  - `1`: Medium
  - `2`: High

## Screenshot
<img width="1738" height="794" alt="image" src="https://github.com/user-attachments/assets/c7e89b1f-35ad-42e8-967f-7753919ea9bd" />


## Author
Boomika S

