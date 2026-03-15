import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import numpy as np

"""Script to create and train an image classification CNN for wildlife species."""

dataset_dir = "data/images"
urls_csv = "data/image_urls.csv"

# make sure the dataset directory exists so we can inspect it later
os.makedirs(dataset_dir, exist_ok=True)

# Try downloading images from CSV
if os.path.isfile(urls_csv):
    import csv
    import requests
    from PIL import Image
    from io import BytesIO
    
    print(f"Downloading images from {urls_csv}...")
    downloaded_count = 0
    
    with open(urls_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row.get('species') or row.get('Species')
            url = row.get('url') or row.get('URL')
            if not sp or not url:
                continue
            
            folder = os.path.join(dataset_dir, sp)
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, f"{sp.lower()}_{downloaded_count}.jpg")
            
            if os.path.exists(fname):
                continue
            
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    # Verify it's a valid image
                    img = Image.open(BytesIO(resp.content))
                    img = img.convert("RGB")
                    img.save(fname)
                    print(f"[OK] Downloaded {fname}")
                    downloaded_count += 1
            except Exception as e:
                print(f"[FAIL] Failed to download {url}: {str(e)[:50]}")

# Check if we have training data
image_count = sum([len(files) for _, _, files in os.walk(dataset_dir)])

if image_count < 10:
    print(f"\n[WARNING] Only {image_count} images found. Creating minimal model...")
    
    # Create a minimal model that can be trained later
    num_classes = len([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d != "__pycache__"])
    num_classes = max(num_classes, 1)
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    os.makedirs("model", exist_ok=True)
    model.save("model/image_model.h5")
    
    # Save class names from directory structure
    classes = sorted([d for d in os.listdir(dataset_dir) 
                     if os.path.isdir(os.path.join(dataset_dir, d)) and d != "__pycache__"])
    with open("model/image_classes.json", "w") as f:
        json.dump(classes, f)
    
    print(f"[OK] Minimal model created with {num_classes} classes: {classes}")
    print("\nTo train the model:")
    print("1. Add more images to data/images/<species>/ folders")
    print("2. Run this script again")
else:
    # Train model with collected images
    print(f"\n[OK] Found {image_count} images. Training model...")
    
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    
    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = len(train_gen.class_indices)
    print(f"Detected {num_classes} species: {list(train_gen.class_indices.keys())}")

    # transfer learning with MobileNetV2
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D

    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64,64,3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        verbose=1
    )

    # fine-tune some layers
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=5, verbose=1)
