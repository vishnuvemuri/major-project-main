import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Step 1: Load Pre-trained VGG16 Model for Feature Extraction
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.output)

# Freeze the layers to prevent retraining
for layer in feature_extractor.layers:
    layer.trainable = False

# Step 2: Prepare Dataset
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Load training data
train_generator = datagen.flow_from_directory(
    r'C:\Users\DELL\Downloads\Vitanmin Project\dataset',  
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=False
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    r'C:\Users\DELL\Downloads\Vitanmin Project\dataset', 
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Step 3: Extract Features
train_features = feature_extractor.predict(train_generator)
train_labels = train_generator.classes

validation_features = feature_extractor.predict(validation_generator)
validation_labels = validation_generator.classes

# Step 4: Train Random Forest Classifier
X_train = train_features.reshape(train_features.shape[0], -1)
X_val = validation_features.reshape(validation_features.shape[0], -1)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, train_labels)

# Save the trained model
joblib.dump(rf_classifier, 'random_forest_vitamin_bnew2.pkl')

# Step 5: Evaluate the Classifier
y_pred = rf_classifier.predict(X_val)
print(classification_report(validation_labels, y_pred, target_names=train_generator.class_indices.keys()))

# Step 6: Predict a New Image's Deficiency
def predict_deficiency(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features using the feature extractor
    features = feature_extractor.predict(img_array)
    features = features.reshape(1, -1)  # Flatten the features for the classifier
    
    # Predict the class label using the pre-trained Random Forest model
    prediction = rf_classifier.predict(features)
    
    # Map the prediction to the corresponding class label
    label_map = {v: k for k, v in train_generator.class_indices.items()}  # Inverse mapping
    class_label = label_map[prediction[0]]
    
    return class_label

