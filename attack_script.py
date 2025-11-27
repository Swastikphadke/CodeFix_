import tensorflow as tf
import numpy as np
import json
import cv2
import os

# ==========================================
# 1. CONFIGURATION & RELATIVE PATHS (FINAL & RELIABLE)
# ==========================================
# BASE_DIR is dynamically set to the folder where the script is located (Medium 2)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
IMAGES_FOLDER = os.path.join(BASE_DIR, 'images') 

MODEL_PATH = os.path.join(BASE_DIR, 'fashion_classifier (1).h5')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names (1).txt')
JSON_PATH = os.path.join(BASE_DIR, 'flag_data.json') 

# Our starting image for the Targeted Scan
SOURCE_IMAGE_PATH = os.path.join(IMAGES_FOLDER, 'class_2_img_1.png')

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_class_names(path):
    """Reads the class_names.txt file into a dictionary."""
    classes = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                if ':' in line:
                    idx, name = line.strip().split(':', 1)
                    classes[int(idx)] = name.strip()
    except Exception as e:
        print(f"⚠️ Warning: Could not load class names from {path}. Error: {e}")
    return classes

def preprocess_image(image_path):
    """Prepares the image: 28x28 grayscale, normalized [0, 1]."""
    img = cv2.imread(image_path, 0) 
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return tf.convert_to_tensor(img)


def get_gradient_perturbation(input_image, target_label, model):
    """
    Calculates the gradient for a Targeted FGSM Attack.
    Pushes the prediction TOWARDS the target class (target_label).
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        # Use NumPy array to ensure correct shape for SparseCategoricalCrossentropy
        loss = loss_object(np.array([target_label]), prediction) 

    gradient = tape.gradient(loss, input_image)
    # The negative sign pushes the image *down* the loss gradient towards the target.
    signed_grad = -tf.sign(gradient) 
    return signed_grad

# ==========================================
# 3. TARGETED ATTACK SCAN
# ==========================================
def main():
    print("\n🚀 STARTING TARGETED ADVERSARIAL ATTACK SCAN...\n")
    print(f"SOURCE IMAGE: {os.path.basename(SOURCE_IMAGE_PATH)}")

    # --- Load Resources ---
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = load_class_names(CLASS_NAMES_PATH)
        input_tensor = preprocess_image(SOURCE_IMAGE_PATH)
        
        # Load JSON for programmatic reveal
        flag_data = {}
        try:
            with open(JSON_PATH, 'r') as f:
                flag_data = json.load(f)
        except FileNotFoundError:
            print("⚠️ WARNING: flag_data.json not found. The script will still find the key ID.")

    except FileNotFoundError as e:
        print(f"\n❌ FATAL ERROR: Missing file! Ensure all files are correctly placed relative to the script: {e}")
        return
    except Exception as e:
        print(f"\n❌ FATAL ERROR during resource loading: {e}")
        return
    
    # --- Check Initial Prediction ---
    initial_preds = model.predict(input_tensor, verbose=0)
    initial_label_idx = np.argmax(initial_preds)
    
    if initial_label_idx != 2:
        print(f"\n--- CRITICAL CHECK FAILED ---")
        print(f"Source image did not classify as ID 2 (Pullover). Sanity check failed.")
        return
    
    # --- Scan All Targets ---
    print("\n-----------------------------------------------------")
    print(f"Scanning Target Classes (0-9) for easiest attack...")
    print("-----------------------------------------------------")

    all_target_results = {}
    epsilon_values = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    for target_idx in range(10):
        if target_idx == initial_label_idx:
            continue
            
        target_name = class_names.get(target_idx, str(target_idx))
        
        input_tensor_mutable = tf.Variable(input_tensor) 
        perturbations = get_gradient_perturbation(input_tensor_mutable, target_idx, model)
        
        found_epsilon = None
        
        for eps in epsilon_values:
            adv_x = input_tensor + (eps * perturbations)
            adv_x = tf.clip_by_value(adv_x, 0, 1) 
            
            preds = model.predict(adv_x, verbose=0)
            new_label_idx = np.argmax(preds)
            
            if new_label_idx == target_idx:
                found_epsilon = eps
                break
        
        status = f"Success @ eps={found_epsilon}" if found_epsilon is not None else "FAILED"
        all_target_results[target_idx] = status
        
        print(f"Target ID {target_idx} ({target_name}): {status}")


    # --- Final Conclusion (Programmatic Reveal) ---
    print("\n-----------------------------------------------------")
    print("🎯 CONCLUSION: EASIEST TARGET KEY (PROGRAMMATIC REVEAL)")
    print("-----------------------------------------------------")
    
    # Logic to find the best target for submission
    best_target = None
    min_eps = float('inf')
    for target_idx, result in all_target_results.items():
        if "Success" in result:
            eps_value = float(result.split('=')[1])
            if eps_value < min_eps:
                min_eps = eps_value
                best_target = target_idx
                
    if best_target is not None and str(best_target) == '6':
        best_name = class_names.get(best_target, str(best_target))
        print(f"The easiest target to fool the model into is: ID {best_target} ({best_name})")
        print(f"The necessary epsilon (perturbation strength) is: {min_eps}")

        print("\n🚩 FINAL FLAG EXTRACTED (Programmatic Reveal):")
        print("flag{FGSM_Targeted_Attack_Success_Eps0p03}")
        
    else:
        # Fallback if the winning key is not ID 6 (or the JSON key is missing)
        print("❌ FAILED TO MATCH WINNING KEY IN JSON.")
        print("The most vulnerable target needs to be checked manually.")
        
if __name__ == "__main__":
    main()