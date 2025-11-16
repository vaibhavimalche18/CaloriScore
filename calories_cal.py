
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model_utils import predict_from_image, load_regression_model, load_nutrition
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# paths to pretrained models (you'll train & place these in /models)
CNN_MODEL_PATH = os.path.join('..','models','cnn_calorie_model.h5')
REG_MODEL_PATH = os.path.join('..','models','regression_calorie_model.pkl')
NUTRITION_CSV = os.path.join('..','datasets','nutrition_data.csv')

# load regression model on startup
try:
    reg_model = load_regression_model(REG_MODEL_PATH)
except Exception:
    reg_model = None

@app.route('/')
def home():
    return jsonify({'status':'ok','message':'Food Calorie Estimator API'})

@app.route('/predict/image', methods=['POST'])
def predict_image():
    # expects form-data with file 'image' and optional 'use_hybrid'
    if 'image' not in request.files:
        return jsonify({'error':'no image part'}), 400
    f = request.files['image']
    if f.filename == '':
        return jsonify({'error':'no selected file'}), 400
    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)

    # CNN prediction
    try:
        cnn_pred = predict_from_image(CNN_MODEL_PATH, filepath)
    except Exception as e:
        return jsonify({'error':'cnn prediction failed', 'detail':str(e)}), 500

    # optionally use regression if class name provided
    use_hybrid = request.form.get('use_hybrid', 'false').lower() == 'true'
    reg_pred = None
    if use_hybrid and reg_model is not None:
        # optional: client can pass nutrients values
        protein = float(request.form.get('protein', 0))
        fat = float(request.form.get('fat', 0))
        carbs = float(request.form.get('carbs', 0))
        fiber = float(request.form.get('fiber', 0))
        try:
            reg_pred = reg_model.predict([[protein, fat, carbs, fiber]])[0]
        except Exception:
            reg_pred = None

    result = {
        'cnn_calories': float(cnn_pred),
        'regression_calories': float(reg_pred) if reg_pred is not None else None,
    }
    if reg_pred is not None:
        hybrid = 0.6 * cnn_pred + 0.4 * reg_pred
        result['hybrid_calories'] = float(hybrid)

    return jsonify(result)

@app.route('/predict/nutrients', methods=['POST'])
def predict_nutrients():
    # expects json with protein,fat,carbs,fiber
    data = request.get_json(force=True)
    protein = float(data.get('protein', 0))
    fat = float(data.get('fat', 0))
    carbs = float(data.get('carbs', 0))
    fiber = float(data.get('fiber', 0))
    if reg_model is None:
        return jsonify({'error':'regression model not available'}), 500
    try:
        pred = reg_model.predict([[protein, fat, carbs, fiber]])[0]
        return jsonify({'predicted_calories': float(pred)})
    except Exception as e:
        return jsonify({'error':'prediction failed', 'detail':str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
