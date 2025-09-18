import gradio as gr
import numpy as np
import tensorflow as tf
import pickle

# Load trained model
model = tf.keras.models.load_model("model.h5")


# Prediction function
def predict_wine_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                         pH, sulphates, alcohol):
    
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                          chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                          pH, sulphates, alcohol]])
    
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]

    quality_map = {0: "Low", 1: "Medium", 2: "High"}  # adjust as per your encoding
    return quality_map[predicted_class]

# Gradio interface using Blocks for layout
with gr.Blocks() as demo:
    gr.Markdown("# 🍷 Wine Quality Prediction")
    gr.Markdown("Predict wine quality (Low, Medium, High) using physicochemical features.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Acidity & Sugar")
            fixed_acidity = gr.Slider(4.0, 16.0, step=0.1, label="Fixed Acidity")
            volatile_acidity = gr.Slider(0.1, 1.5, step=0.01, label="Volatile Acidity")
            citric_acid = gr.Slider(0.0, 1.0, step=0.01, label="Citric Acid")
            residual_sugar = gr.Slider(0.9, 15.0, step=0.1, label="Residual Sugar")
        
        with gr.Column(scale=1):
            gr.Markdown("### Sulfur & Density")
            chlorides = gr.Slider(0.01, 0.2, step=0.001, label="Chlorides")
            free_sulfur_dioxide = gr.Slider(1, 72, step=1, label="Free Sulfur Dioxide")
            total_sulfur_dioxide = gr.Slider(6, 289, step=1, label="Total Sulfur Dioxide")
            density = gr.Slider(0.990, 1.004, step=0.0001, label="Density")
        
        with gr.Column(scale=1):
            gr.Markdown("### Other Chemistry")
            pH = gr.Slider(2.5, 4.0, step=0.01, label="pH")
            sulphates = gr.Slider(0.3, 2.0, step=0.01, label="Sulphates")
            alcohol = gr.Slider(8.0, 15.0, step=0.1, label="Alcohol")
    
    predict_btn = gr.Button("Predict Quality")
    output = gr.Label(label="Predicted Wine Quality")
    
    # Button click triggers prediction
    predict_btn.click(
        fn=predict_wine_quality,
        inputs=[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                pH, sulphates, alcohol],
        outputs=output
    )

demo.launch()
