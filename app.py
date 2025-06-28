from flask import Flask, render_template, request, redirect, url_for
import os
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__, template_folder='flask/templates', static_folder='flask/static')
UPLOAD_FOLDER = os.path.join('flask', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model('cnn.keras')  # ← Change if needed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Preprocess image
        img = Image.open(filepath).convert('RGB')  
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  
        print("Model input shape expected:", model.input_shape)
        print("Prepared image shape:", img_array.shape)

        # Predict
        prediction = model.predict(img_array)
        op = ["arecaceae",'anadenanthera','arrabidea','cecropia','chromolaena','combretum','croton','dipteryx','eucalipto','faramea','hyptis','mabea','matayba','mimosa','myrcia','protium','qualea','schinus','senegalia','sergenia','syagrus','tridax','urochloa']
        result = op[np.argmax(prediction)]

        return render_template('prediction.html', prediction=result)

    except Exception as e:
        return f"Error during prediction: {str(e)}"

@app.route('/logout')
def logout():
    return render_template('logout.html')

if __name__ == '__main__':
    app.run(debug=True)
