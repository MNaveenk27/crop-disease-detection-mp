import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json 
from flask import Flask, render_template, request, url_for 
from keras.saving import load_model
from tensorflow.keras.preprocessing import image 
import numpy as np 
import os 
import cv2 
from werkzeug.utils import secure_filename 
import logging 
app = Flask(__name__) 
model = load_model('model/final_crop_disease_model.h5') 
UPLOAD_FOLDER = 'static/uploads' 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load treatments from JSON file
with open('treatments.json', 'r') as f:
    TREATMENTS = json.load(f)

class_names = [ 
'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy', 
'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
    'Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)_Common_rust', 
    'Corn_(maize)Northern_Leaf_Blight', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy', 
    'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight', 
    'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy', 
    'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy', 
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato__Late_blight', 
    'Tomato__Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Spider_mites_Twospotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato__Tomato_mosaic_virus','Tomato___healthy' 
] 
 
DISEASE_COLOR_RANGES = { 
    'default': { 
        'lower': np.array([20, 20, 20]), 
        'upper': np.array([60, 255, 255]) 
    }, 
    'Apple__Apple_scab': { 
        'lower': np.array([15, 40, 40]), 
        'upper': np.array([35, 255, 255]) 
    }, 
    'Apple_Black_rot': { 
        'lower': np.array([5, 60, 30]), 
        'upper': np.array([25, 255, 200]) 
    }, 
    'Apple_Cedar_apple_rust': { 
        'lower': np.array([10, 100, 100]), 
        'upper': np.array([25, 255, 255]) 
    }, 
    'Tomato_Early_blight': { 
        'lower': np.array([10, 50, 50]), 
        'upper': np.array([40, 255, 255]) 
    },  
    'Tomato__Late_blight': { 
        'lower': np.array([15, 45, 45]), 
        'upper': np.array([35, 255, 255]) 
    }, 
    'Tomato__Leaf_Mold': { 
        'lower': np.array([25, 40, 40]), 
        'upper': np.array([50, 255, 255]) 
    }, 
    'Tomato_Septoria_leaf_spot': { 
        'lower': np.array([20, 60, 60]), 
        'upper': np.array([40, 255, 255]) 
    }, 
    'Tomato__Spider_mites_Two-spotted_spider_mite': { 
        'lower': np.array([0, 60, 60]), 
        'upper': np.array([20, 255, 255]) 
    }, 
    'Tomato__Target_Spot': { 
        'lower': np.array([10, 70, 50]), 
        'upper': np.array([30, 255, 255]) 
    }, 
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus': { 
        'lower': np.array([25, 100, 100]), 
        'upper': np.array([35, 255, 255]) 
    }, 
    'Tomato__Tomato_mosaic_virus': { 
        'lower': np.array([30, 100, 100]), 
        'upper': np.array([60, 255, 255]) 
    }, 
    'Tomato__Bacterial_spot': { 
        'lower': np.array([15, 70, 70]), 
        'upper': np.array([35, 255, 255]) 
    }, 
    'Potato__Early_blight': { 
        'lower': np.array([15, 60, 60]), 
        'upper': np.array([35, 255, 255]) 
    }, 
    'Potato__Late_blight': { 
        'lower': np.array([10, 45, 45]), 
        'upper': np.array([30, 255, 255]) 
    }, 
    'Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot': { 
        'lower': np.array([20, 60, 60]), 
        'upper': np.array([40, 255, 255]) 
    }, 
    'Corn(maize)_Common_rust': { 
        'lower': np.array([10, 60, 60]), 
        'upper': np.array([30, 255, 255]) 
    }, 
    'Corn_(maize)Northern_Leaf_Blight': { 
        'lower': np.array([15, 50, 50]), 
        'upper': np.array([35, 255, 255]) 
    }, 
    'Cherry(including_sour)_Powdery_mildew': { 
        'lower': np.array([0, 0, 200]), 
        'upper': np.array([180, 30, 255]) 
    }, 
    'Grape_Black_rot': { 
        'lower': np.array([20, 40, 40]), 
        'upper': np.array([40, 255, 255]) 
    }, 
    'Grape_Esca(Black_Measles)': { 
        'lower': np.array([15, 40, 40]), 
        'upper': np.array([35, 255, 255]) 
    }, 
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': { 
        'lower': np.array([20, 50, 50]), 
        'upper': np.array([40, 255, 255]) 
    }, 
    'Orange__Haunglongbing(Citrus_greening)': { 
        'lower': np.array([25, 100, 100]), 
        'upper': np.array([35, 255, 255]) 
    }, 
    'Peach__Bacterial_spot': { 
        'lower': np.array([10, 60, 60]), 
        'upper': np.array([30, 255, 255]) 
    }, 
    'Pepper,bell_Bacterial_spot': { 
        'lower': np.array([10, 70, 70]), 
        'upper': np.array([30, 255, 255]) 
    }, 
    'Squash__Powdery_mildew': { 
        'lower': np.array([0, 0, 200]), 
        'upper': np.array([180, 30, 255]) 
    }, 
    'Strawberry_Leaf_scorch': { 
        'lower': np.array([15, 70, 70]), 
        'upper': np.array([35, 255, 255]) 
    } 
} 
# Function to separate treatments by type (organic vs chemical)
def categorize_treatments():
    organic_treatments = {}
    chemical_treatments = {}
    
    for disease, treatments in TREATMENTS.items():
        # Skip entries that are already dictionaries (severity-based treatments)
        if isinstance(treatments, dict):
            continue
            
        organic_list = []
        chemical_list = []
        
        for treatment in treatments:
            # Check if treatment is organic based on keywords
            if any(keyword in treatment.lower() for keyword in 
                  ['neem', 'compost tea', 'copper', 'sulfur', 'biological', 
                   'remove', 'prune', 'rotate', 'resistant', 'spacing', 'mulch', 
                   'monitor', 'sticky trap', 'predator', 'disinfect', 'baking soda']):
                organic_list.append(treatment)
            # Check if treatment is chemical based on keywords
            elif any(keyword in treatment.lower() for keyword in 
                    ['fungicide', 'insecticide', 'bactericide', 'chlorothalonil', 
                     'mancozeb', 'propiconazole', 'metalaxyl', 'imidacloprid', 
                     'abamectin', 'captan', 'myclobutanil', 'streptomycin', 'oxytetracycline']):
                chemical_list.append(treatment)
            # If not clearly categorized, add to both lists
            else:
                organic_list.append(treatment)
                chemical_list.append(treatment)
        
        organic_treatments[disease] = organic_list
        chemical_treatments[disease] = chemical_list
    
    return organic_treatments, chemical_treatments

# Get pre-categorized treatments
ORGANIC_TREATMENTS, CHEMICAL_TREATMENTS = categorize_treatments()

def get_treatment_by_severity(disease_name, severity, treatment_type='all'):
    """
    Returns treatment recommendations based on disease, severity, and treatment type.
    Since treatments.json does not have severity-based treatments, severity is currently ignored.
    treatment_type can be 'organic', 'chemical', or 'all'.
    """
    if treatment_type == 'organic':
        return ORGANIC_TREATMENTS.get(disease_name, [])
    elif treatment_type == 'chemical':
        return CHEMICAL_TREATMENTS.get(disease_name, [])
    else:
        # Return combined list without duplicates
        organic = ORGANIC_TREATMENTS.get(disease_name, [])
        chemical = CHEMICAL_TREATMENTS.get(disease_name, [])
        combined = list(dict.fromkeys(organic + chemical))
        return combined

def calculate_severity(image_path, disease_name): 
    try: 
        image = cv2.imread(image_path) 
        if image is None: 
            return "Medium", 15.0 
        resized = cv2.resize(image, (256, 256)) 
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV) 
        color_range = DISEASE_COLOR_RANGES.get(disease_name,DISEASE_COLOR_RANGES['default']) 
        lower_bound = color_range['lower'] 
        upper_bound = color_range['upper'] 
        mask = cv2.inRange(hsv, lower_bound, upper_bound) 
        damaged_pixels = cv2.countNonZero(mask) 
        total_pixels = mask.size 
        damage_percent = (damaged_pixels / total_pixels) * 100 

        if damage_percent < 10: 
            severity = "Low" 
        elif 10 <= damage_percent < 30: 
            severity = "Medium" 
        else: 
            severity = "High" 
        return severity, damage_percent 
    except Exception as e: 
        app.logger.error(f"Error in severity calculation: {e}", exc_info=True) 
        return "Medium", 15.0 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) 
def predict(): 
    try: 
        if 'leaf_image' not in request.files: 
            return 'No file uploaded.', 400 
        file = request.files['leaf_image'] 
        if file.filename == '': 
            return 'No selected file.', 400 
        area_size = float(request.form.get('area_size', 100)) 
        treatment_type = request.form.get('treatment_type', 'all') 
        filename = secure_filename(file.filename) 
        filepath = os.path.join(UPLOAD_FOLDER, filename) 
        file.save(filepath) 

        img = cv2.imdecode(np.frombuffer(open(filepath, 'rb').read(), np.uint8), cv2.IMREAD_COLOR) 
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        lower_brown = np.array([0, 50, 20]) 
        upper_brown = np.array([30, 255, 255]) 
        brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown) 
        brown_pixels = cv2.countNonZero(brown_mask) 
        total_pixels = img.shape[0] * img.shape[1] 
        brown_ratio = brown_pixels / total_pixels 
        is_leaf = brown_ratio > 0.005 
        if not is_leaf: 
            return render_template('index.html', error_message="The uploaded image does not appear to be a leaf. Please upload a valid leaf image.") 
        img_for_model = image.load_img(filepath, target_size=(224, 224)) 
        img_array = image.img_to_array(img_for_model) / 255.0 
        img_array = np.expand_dims(img_array, axis=0) 
        prediction = model.predict(img_array) 
        predicted_class = class_names[np.argmax(prediction)] 
        severity, damage_percent = calculate_severity(filepath, predicted_class) 
        treatment_recommendations = get_treatment_by_severity(predicted_class, severity, treatment_type)
        if not predicted_class.endswith('healthy') and area_size > 0: 
            pass 
        image_url = url_for('static', filename='uploads/' + filename) 
        # Format the severity-based treatment recommendations as HTML
        treatment_html = f"<h3>Treatment Recommendations for {predicted_class} (Severity: {severity}, {damage_percent:.1f}% damage)</h3><ul>"
        for treatment in treatment_recommendations:
            treatment_html += f"<li>{treatment}</li>"
        treatment_html += "</ul>"
        # Get severity-based treatment recommendations separately for organic and chemical
        organic_treatments = get_treatment_by_severity(predicted_class, severity, 'organic')
        chemical_treatments = get_treatment_by_severity(predicted_class, severity, 'chemical')

        # Format the severity and treatment recommendations as HTML
        treatment_html = f"<h3>Treatment Recommendations for {predicted_class} (Severity: {severity}, {damage_percent:.1f}% damage)</h3>"
        if treatment_type == 'organic':
            treatment_html += "<h4>Organic Treatments:</h4><ul>"
            for treatment in organic_treatments:
                treatment_html += f"<li>{treatment}</li>"
            treatment_html += "</ul>"
        elif treatment_type == 'chemical':
            treatment_html += "<h4>Chemical Treatments:</h4><ul>"
            for treatment in chemical_treatments:
                treatment_html += f"<li>{treatment}</li>"
            treatment_html += "</ul>"
        else:  # Show both
            treatment_html += "<h4>Organic Treatments:</h4><ul>"
            for treatment in organic_treatments:
                treatment_html += f"<li>{treatment}</li>"
            treatment_html += "</ul>"
            treatment_html += "<h4>Chemical Treatments:</h4><ul>"
            for treatment in chemical_treatments:
                treatment_html += f"<li>{treatment}</li>"
            treatment_html += "</ul>"

        # Add message for healthy leaves
        healthy_classes = [cls for cls in class_names if 'healthy' in cls.lower()]
        if predicted_class in healthy_classes:
            # Use severity-based treatments for healthy leaves if available
            if predicted_class in TREATMENTS and isinstance(TREATMENTS[predicted_class], dict) and severity in TREATMENTS[predicted_class]:
                treatments = TREATMENTS[predicted_class][severity]
                treatment_html = f"<h3>Treatment Recommendations for {predicted_class} (Severity: {severity})</h3><ul>"
                for treatment in treatments:
                    treatment_html += f"<li>{treatment}</li>"
                treatment_html += "</ul>"
                # Add additional recommendations as requested
                treatment_html += "<h4>Recommendations:</h4><ul>"
                treatment_html += "<li>Maintain regular monitoring (weekly)</li>"
                treatment_html += "<li>Ensure proper irrigation and avoid wet foliage</li>"
                treatment_html += "<li>Use organic mulch to retain moisture</li>"
                treatment_html += "<li>Follow a balanced fertilizer schedule</li>"
                treatment_html += "</ul>"
            else:
                treatment_html = "<h3>Plant Status: Healthy</h3>"
                treatment_html += "<p>Your plant appears to be healthy! Here are some maintenance tips:</p><ul>"
                treatment_html += "<li>Ensure it is getting sufficient sunlight</li>"
                treatment_html += "<li>Water regularly but avoid overwatering</li>"
                treatment_html += "<li>Keep the soil well-drained</li>"
                treatment_html += "<li>Continue regular inspections for early disease detection</li>"
                treatment_html += "</ul>"
        # Calculate estimated treatment amounts based on area size
        if not predicted_class.endswith('healthy') and area_size > 0:
            treatment_html += f"<h4>Treatment Quantities (for {area_size} sq. meters):</h4><ul>"
            
            # Combine organic and chemical treatments for quantity calculation
            combined_treatments = organic_treatments + chemical_treatments
            
            if any("fungicide" in t.lower() for t in combined_treatments):
                fungicide_amount = area_size * 0.3  # 0.3 liters per sq meter
                treatment_html += f"<li>Fungicide solution needed: approximately {fungicide_amount:.1f} liters</li>"
            
            if any("copper" in t.lower() for t in combined_treatments):
                copper_amount = area_size * 3  # 3g per sq meter
                treatment_html += f"<li>Copper product needed: approximately {copper_amount:.1f} grams</li>"
            
            if any("neem" in t.lower() for t in combined_treatments):
                neem_amount = area_size * 0.2  # 0.2 liters per sq meter
                treatment_html += f"<li>Neem oil solution needed: approximately {neem_amount:.1f} liters</li>"
            
            treatment_html += "</ul>"
            
            # Additional tips based on severity
            treatment_html += "<h4>Additional recommendations:</h4><ul>"
            if severity == "Low":
                treatment_html += "<li>Consider preventive measures first before using chemical treatments</li>"
                treatment_html += "<li>Regular monitoring every 3-5 days is recommended</li>"
            elif severity == "Medium":
                treatment_html += "<li>Begin treatment immediately</li>"
                treatment_html += "<li>Consider isolating affected plants if possible</li>"
                treatment_html += "<li>Monitor daily for disease progression</li>"
            else:  # High severity
                treatment_html += "<li>Urgent treatment required</li>"
                treatment_html += "<li>Consider removing and destroying severely affected plants</li>"
                treatment_html += "<li>Treat surrounding plants preventively</li>"
                treatment_html += "<li>Daily monitoring is essential</li>"
            treatment_html += "</ul>"
        return render_template( 
            'index.html',  
            prediction=predicted_class,  
            image_path=image_url, 
            treatment=treatment_html, 
            area_size=area_size, 
            treatment_type=treatment_type, 
            severity=severity 
        ) 
    except Exception as e: 
        app.logger.error(f"Error during prediction: {e}", exc_info=True) 
        return f"An error occurred during prediction: {e}", 500 
if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000, debug=True)
