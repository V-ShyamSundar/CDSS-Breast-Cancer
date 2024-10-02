import os
import pickle
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tqdm.keras import TqdmCallback  # For the progress bar
from tensorflow.keras.callbacks import ModelCheckpoint  # For saving model checkpoints

# Load CSV files for metadata (train and test)
train_metadata = pd.read_csv('Train-Set.csv')
test_metadata = pd.read_csv('Test-Set.csv')

# Extract the patient IDs from the metadata
train_metadata['patient_id'] = train_metadata['patient_id'].astype(str)
test_metadata['patient_id'] = test_metadata['patient_id'].astype(str)

# Folder paths
base_dir = 'Dataset'
cc_train_dir = os.path.join(base_dir, 'CC_full')
cc_test_dir = os.path.join(base_dir, 'CC_full_test')
mlo_train_dir = os.path.join(base_dir, 'MLO_full')
mlo_test_dir = os.path.join(base_dir, 'MLO_full_test')

# Function to load images and labels from the dataset folders and match with metadata
def load_images_and_labels_with_metadata(directory, metadata_df):
    images = []
    labels = []
    matched_metadata = []

    for label in ['benign', 'benign wout', 'malignant']:
        label_dir = os.path.join(directory, label)
        for file_name in os.listdir(label_dir):
            patient_id = file_name.removesuffix(".jpg")
            if patient_id in metadata_df['patient_id'].values:
                img_path = os.path.join(label_dir, file_name)
                img = cv2.imread(img_path, 0)  # Load grayscale image
                img_resized = cv2.resize(img, (1024, 1024))
                images.append(img_resized)

                if label in ['benign', 'benign wout']:
                    labels.append('benign')  # Treat both as 'benign'
                else:
                    labels.append(label)  # Keep 'malignant'

                matched_metadata.append(metadata_df[metadata_df['patient_id'] == patient_id].iloc[0])

    return np.array(images), np.array(labels), pd.DataFrame(matched_metadata)

# Load data and metadata for training and testing
cc_train_images, cc_train_labels, cc_train_metadata = load_images_and_labels_with_metadata(cc_train_dir, train_metadata)
mlo_train_images, mlo_train_labels, mlo_train_metadata = load_images_and_labels_with_metadata(mlo_train_dir, train_metadata)
cc_test_images, cc_test_labels, cc_test_metadata = load_images_and_labels_with_metadata(cc_test_dir, test_metadata)
mlo_test_images, mlo_test_labels, mlo_test_metadata = load_images_and_labels_with_metadata(mlo_test_dir, test_metadata)

# Combine CC and MLO images and metadata
train_images = np.concatenate((cc_train_images, mlo_train_images))
train_labels = np.concatenate((cc_train_labels, mlo_train_labels))
test_images = np.concatenate((cc_test_images, mlo_test_images))
test_labels = np.concatenate((cc_test_labels, mlo_test_labels))

train_metadata_combined = pd.concat([cc_train_metadata, mlo_train_metadata])
test_metadata_combined = pd.concat([cc_test_metadata, mlo_test_metadata])

# Encode labels into numeric values
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Reshape images to add the channel dimension (grayscale: 1 channel)
train_images = train_images.reshape(train_images.shape[0], 1024, 1024, 1)
test_images = test_images.reshape(test_images.shape[0], 1024, 1024, 1)

# Normalize pixel values
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to categorical
train_labels_categorical = to_categorical(train_labels_encoded, num_classes=2)
test_labels_categorical = to_categorical(test_labels_encoded, num_classes=2)

# Metadata preprocessing
metadata_features = ['breast_density', 'mass_shape', 'mass_margins', 'assessment', 'subtlety']
combined_metadata = pd.concat([train_metadata_combined, test_metadata_combined])
combined_metadata['mass_shape'] = label_encoder.fit_transform(combined_metadata['mass_shape'])
combined_metadata['mass_margins'] = label_encoder.fit_transform(combined_metadata['mass_margins'])

train_metadata_combined['mass_shape'] = combined_metadata['mass_shape'][:len(train_metadata_combined)]
train_metadata_combined['mass_margins'] = combined_metadata['mass_margins'][:len(train_metadata_combined)]
test_metadata_combined['mass_shape'] = combined_metadata['mass_shape'][len(train_metadata_combined):]
test_metadata_combined['mass_margins'] = combined_metadata['mass_margins'][len(train_metadata_combined):]

train_metadata_features = train_metadata_combined[metadata_features].values
test_metadata_features = test_metadata_combined[metadata_features].values

# Standardize numeric columns
scaler = StandardScaler()
train_metadata_features = scaler.fit_transform(train_metadata_features)
test_metadata_features = scaler.transform(test_metadata_features)

# Create dataset using tf.data.Dataset
def create_dataset(images, metadata, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(({"image": images, "metadata": metadata}, labels))
    dataset = dataset.batch(batch_size).shuffle(len(images)).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create train and test datasets
batch_size = 16
train_dataset = create_dataset(train_images, train_metadata_features, train_labels_categorical, batch_size)
test_dataset = create_dataset(test_images, test_metadata_features, test_labels_categorical, batch_size)

# CNN model for image input
def create_model():
    # Image input layer
    image_input = Input(shape=(1024, 1024, 1), name="image")
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)

    # Dense layers for metadata input
    metadata_input = Input(shape=(train_metadata_features.shape[1],), name="metadata")
    y = layers.Dense(64, activation='relu')(metadata_input)

    # Concatenate image features and metadata features
    combined = layers.concatenate([x, y])

    # Fully connected layers after concatenation
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dense(2, activation='softmax')(z)  # 2 classes (benign and malignant)

    # Create the multi-input model
    model = Model(inputs=[image_input, metadata_input], outputs=z)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Define the filepaths for model saving/loading
final_model_filepath = 'CDSS_CNN_final.keras'

# Check if history exists (the training_history.pkl file)
history_path = 'training_history.pkl'

# Check if a previous model exists to resume training
if os.path.exists(final_model_filepath):
    print(f"Loading model from {final_model_filepath}...")
    model = load_model(final_model_filepath)  # Load the entire model
    print("Model loaded successfully, resuming training.")
    try:
        # Load the saved training history using pickle
        with open(history_path, 'rb') as file:
            history_data = pickle.load(file)
        initial_epoch = history_data['epoch'][-1]
        print(f"Resuming training from epoch {initial_epoch}")
    except Exception as e:
        print(f"Failed to load history, starting from epoch 0. Error: {e}")
        initial_epoch = 0
        history_data = None
else:
    print("No saved model found, creating a new model.")
    model = create_model()
    initial_epoch = 0
    history_data = None


# Train the model, resuming from previous training (if any)
total_epochs = initial_epoch + 20
history = model.fit(train_dataset,
          validation_data=test_dataset,
          epochs=total_epochs,
          initial_epoch=initial_epoch,
          callbacks=[TqdmCallback(verbose=1)])

# Evaluate the model on the test set (both images and metadata)
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Save the final model after training
model.save(final_model_filepath)

# Prepare history data for saving, including epochs
if history_data:
    for key in history.history:
        history_data[key].extend(history.history[key])
    history_data['epoch'].extend(list(range(initial_epoch+1, total_epochs+1)))
else:
    history_data = history.history
    history_data['epoch'] = list(range(initial_epoch, total_epochs))

# Save the training history using pickle
with open(history_path, 'wb') as file:
    pickle.dump(history_data, file)

print(f"Model training complete, saved to {final_model_filepath}")

