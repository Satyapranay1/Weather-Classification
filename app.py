from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('model1.h5')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="No file uploaded")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', result="No selected file")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            prediction = model.predict(img_array)
            weather_classes = ['Cloudy', 'Rainy','Shine', 'Sunrise']
            predicted_class = weather_classes[np.argmax(prediction)]
            
            return render_template('index.html', result=f'Predicted Weather: {predicted_class}')
        except Exception as e:
            return render_template('index.html', result=f"Error during prediction: {e}")
        finally:
            os.remove(filepath)  # Remove the file after prediction
    else:
        return render_template('index.html', result="Invalid file format. Please upload a .jpg, .jpeg, or .png file.")

if __name__ == '__main__':
    app.run(debug=True)
