#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, time
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import keras_hub

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import l2


# In[3]:


print("TensorFlow:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices[0])
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[5]:


IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_CLASSES = 45   # RESISC45 has 45 scene classes
EPOCHS      = 30
AUTOTUNE    = tf.data.AUTOTUNE


# In[6]:


# RESISC45 – 45 scene classes, 700 images/class, 256×256 RGB
# Images are already 256×256 so no resize step needed in preprocessing

(train_raw, val_raw, test_raw), info_ds = tfds.load(
    'resisc45',
    split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
    shuffle_files=False,   # must be False with percentage splits
    as_supervised=True,
    with_info=True
)

CLASS_NAMES = info_ds.features['label'].names
total       = info_ds.splits['train'].num_examples

print(f'Classes          : {NUM_CLASSES}')
print(f'Class names      : {CLASS_NAMES}')
print(f'Train (~70%)     : ~{int(total * 0.70)}')
print(f'Val   (~15%)     : ~{int(total * 0.15)}')
print(f'Test  (~15%)     : ~{int(total * 0.15)}')
print(f'Images per class : ~{int(total * 0.70) // NUM_CLASSES} (train)')

for image, label in train_raw.take(3):
    print(f'Label: {label.numpy()} → {CLASS_NAMES[label.numpy()]}, Image shape: {image.shape}')


# In[7]:


builder = tfds.builder('resisc45')
for split_name, split_info in builder.info.splits.items():
    print(f'{split_name}: {split_info.num_shards} shards, {split_info.num_examples} examples')


# In[10]:


fig = tfds.show_examples(train_raw, info_ds, rows=1, cols=5)
fig = tfds.show_examples(val_raw,   info_ds, rows=1, cols=5)
fig = tfds.show_examples(test_raw,  info_ds, rows=1, cols=5)


# In[11]:


# RESISC45 images are already 256×256 — no tf.image.resize needed

def preprocess_train_ds(image, label):
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])  # 256→224 random crop
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)   # valid: satellite has no fixed 'up'
    image = tf.image.random_brightness(image, 0.2)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

def preprocess_ds(image, label):
    image = tf.image.central_crop(image, 0.875)        # 256 × 0.875 = 224 — deterministic
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label


train_ds = (
    train_raw
    .cache()
    .map(preprocess_train_ds, num_parallel_calls=AUTOTUNE)
    .shuffle(2000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_raw
    .map(preprocess_ds, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_raw
    .map(preprocess_ds, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# In[12]:


resnet50_base = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(IMG_SIZE, IMG_SIZE, 3))
resnet50_base.trainable = False

print(f"{'Layer (type)':<25} | {'Output Shape':<20} | {'Params':<10}")
print("-" * 65)

# First 10 layers of ResNet_50
for layer in resnet50_base.layers[:10]:
    name = layer.name
    shape = str(layer.output.shape)
    params = f"{layer.count_params():,}"
    print(f"{name:<25} | {shape:<20} | {params:<10}")




# In[13]:


# --- MODEL 1: ResNet_50 base ---

model_base = tf.keras.Sequential()
model_base.add(tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model_base.add(resnet50_base)                                # spectial_features
model_base.add(tf.keras.layers.GlobalAveragePooling2D())                      # pooled_features
model_base.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

# img_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# spatial_features = resnet50_model(img_input, training= False)
# pooled_features = tf.keras.layers.GlobalAveragePooling2D()(spatial_features)
# class_probabilities = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(pooled_features)
# flower_model_simple = tf.keras.Model(img_input, class_probabilities)

model_base.compile(
        optimizer= 'adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
)

model_base.summary()


# In[14]:


print('BATCH_SIZE:', BATCH_SIZE)
print('NUM_CLASSES:', NUM_CLASSES)

# Check one batch shape
for x, y in train_ds.take(1):
    print('Batch shape:', x.shape)
    print('Sample labels:', y.numpy()[:10])
    print('Sample class names:', [CLASS_NAMES[l] for l in y.numpy()[:5]])


# In[1]:


print("--- Training ResNet-50 base ---")
t0 = time.time()
resnet = model_base.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
time_taken = time.time() - t0
best_accuracy = max(resnet.history['val_accuracy']) * 100
print(f"\nBest Val Accuracy: {best_accuracy:.2f}%")
print(f"Training Time:     {time_taken/60:.1f} min")


# In[ ]:


import matplotlib.pyplot as plt

# 1. Set up a clean canvas for two side-by-side plots
plt.figure(figsize=(14, 5))

# --- PLOT 1: Accuracy Comparison ---
plt.subplot(1, 2, 1)
# Plot training & validation for the Original Simple Model
plt.plot(resnet.history['accuracy'], label='Simple - Train Acc', color='lightblue', linestyle='--')
plt.plot(resnet.history['val_accuracy'], label='Simple - Val Acc', color='blue')


plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)

# --- PLOT 2: Loss Comparison ---
plt.subplot(1, 2, 2)
# Plot training & validation loss for the Original Simple Model
plt.plot(resnet.history['loss'], label='Simple - Train Loss', color='lightblue', linestyle='--')
plt.plot(resnet.history['val_loss'], label='Simple - Val Loss', color='blue')


plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Render the layout cleanly
plt.tight_layout()
plt.show()


# In[ ]:


# --- MODEL 2: ResNet_50 with head: Deep Neural Network  ---

model_DNN = tf.keras.Sequential()
model_DNN.add(tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model_DNN.add(resnet50_base)
model_DNN.add(tf.keras.layers.GlobalAveragePooling2D())

model_DNN.add(tf.keras.layers.Dense(1024))
model_DNN.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
# model_DNN.add(tf.keras.layers.Dropout(0.3))

model_DNN.add(tf.keras.layers.Dense(512))
model_DNN.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
# model_DNN.add(tf.keras.layers.Dropout(0.3))

model_DNN.add(tf.keras.layers.Dense(256))
model_DNN.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
# model_DNN.add(tf.keras.layers.Dropout(0.3))

model_DNN.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model_DNN.compile(
    optimizer= 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_DNN.summary()

# img_input_2 = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# spatial_features_2 = resnet50_model(img_input_2, training= False)
# pooled_features_2 = tf.keras.layers.GlobalAveragePooling2D()(spatial_features_2)

# dense_layer_01 = tf.keras.layers.Dense(1024)(pooled_features_2)
# dense_layer_01 = tf.keras.layers.LeakyReLU(negative_slope=0.01)(dense_layer_01)
# dense_layer_01 = tf.keras.layers.Dropout(0.3)(dense_layer_01)

# dense_layer_02 = tf.keras.layers.Dense(512)(dense_layer_01)
# dense_layer_02 = tf.keras.layers.LeakyReLU(negative_slope=0.01)(dense_layer_02)
# dense_layer_02 = tf.keras.layers.Dropout(0.3)(dense_layer_02)

# dense_layer_03 = tf.keras.layers.Dense(256)(dense_layer_02)
# dense_layer_03 = tf.keras.layers.LeakyReLU(negative_slope=0.01)(dense_layer_03)
# dense_layer_03 = tf.keras.layers.Dropout(0.3)(dense_layer_03)

# class_probabilities_2 = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(dense_layer_03)

# flower_model_FC = tf.keras.Model(img_input_2, class_probabilities_2)
# flower_model_FC.compile(
#     optimizer= tf.keras.optimizers.Adam(1e-3),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )


# In[ ]:


print("--- Training ResNet_50 with DNN ---")
t0 = time.time()
resnet_DNN = model_DNN.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
time_taken = time.time() - t0
best_accuracy = max(resnet_DNN.history['val_accuracy']) * 100
print(f"\nBest Val Accuracy: {best_accuracy:.2f}%")
print(f"Training Time:     {time_taken/60:.1f} min")


# In[ ]:


import matplotlib.pyplot as plt

# 1. Set up a clean canvas for two side-by-side plots
plt.figure(figsize=(14, 5))

# --- PLOT 1: Accuracy Comparison ---
plt.subplot(1, 2, 1)
# Plot training & validation for the Original Simple Model
plt.plot(resnet_history.history['accuracy'], label='Simple - Train Acc', color='lightblue', linestyle='--')
plt.plot(resnet_history.history['val_accuracy'], label='Simple - Val Acc', color='blue')

# Plot training & validation for your New FC Leaky-ReLU Model
plt.plot(resnet_history_FC.history['accuracy'], label='FC Leaky - Train Acc', color='lightgreen', linestyle='--')
plt.plot(resnet_history_FC.history['val_accuracy'], label='FC Leaky - Val Acc', color='green')

plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)

# --- PLOT 2: Loss Comparison ---
plt.subplot(1, 2, 2)
# Plot training & validation loss for the Original Simple Model
plt.plot(resnet_history.history['loss'], label='Simple - Train Loss', color='lightblue', linestyle='--')
plt.plot(resnet_history.history['val_loss'], label='Simple - Val Loss', color='blue')

# Plot training & validation loss for your New FC Leaky-ReLU Model
plt.plot(resnet_history_FC.history['loss'], label='FC Leaky - Train Loss', color='lightgreen', linestyle='--')
plt.plot(resnet_history_FC.history['val_loss'], label='FC Leaky - Val Loss', color='green')

plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Render the layout cleanly
plt.tight_layout()
plt.show()


# In[ ]:


# --- MODEL 3: ResNet_50 & DNN : Droupouts  ---

resnet50_base.trainable = False
model_DNN_DO = tf.keras.Sequential()
model_DNN_DO.add(tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model_DNN_DO.add(resnet50_base)
model_DNN_DO.add(tf.keras.layers.GlobalAveragePooling2D())

model_DNN_DO.add(tf.keras.layers.Dense(1024))
model_DNN_DO.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
model_DNN_DO.add(tf.keras.layers.Dropout(0.3))

model_DNN_DO.add(tf.keras.layers.Dense(512))
model_DNN_DO.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
model_DNN_DO.add(tf.keras.layers.Dropout(0.3))

model_DNN_DO.add(tf.keras.layers.Dense(256))
model_DNN_DO.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
model_DNN_DO.add(tf.keras.layers.Dropout(0.3))

model_DNN_DO.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model_DNN_DO.compile(
    optimizer= 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_DNN_DO.summary()


# In[ ]:


print("--- Training MODEL 3: ResNet_50 & DNN : Droupouts  ---")
t0 = time.time()
resnet_DNN_DO = model_DNN_DO.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
time_taken = time.time() - t0
best_accuracy = max(resnet_DNN_DO.history['val_accuracy']) * 100
print(f"\nBest Val Accuracy: {best_accuracy:.2f}%")
print(f"Training Time:     {time_taken/60:.1f} min")


# In[ ]:


# --- MODEL 4: ResNet & DNN : Batch Normalization + Dropout ---

resnet50_base.trainable = False
model_DNN_DO_BN = tf.keras.Sequential()
model_DNN_DO_BN.add(tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model_DNN_DO_BN.add(resnet50_base)
model_DNN_DO_BN.add(tf.keras.layers.GlobalAveragePooling2D())

model_DNN_DO_BN.add(tf.keras.layers.Dense(1024, use_bias=False))
model_DNN_DO_BN.add(tf.keras.layers.BatchNormalization())
model_DNN_DO_BN.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
model_DNN_DO_BN.add(tf.keras.layers.Dropout(0.4))

model_DNN_DO_BN.add(tf.keras.layers.Dense(512, use_bias=False))
model_DNN_DO_BN.add(tf.keras.layers.BatchNormalization())
model_DNN_DO_BN.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
model_DNN_DO_BN.add(tf.keras.layers.Dropout(0.4))

model_DNN_DO_BN.add(tf.keras.layers.Dense(256, use_bias=False))
model_DNN_DO_BN.add(tf.keras.layers.BatchNormalization())
model_DNN_DO_BN.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
model_DNN_DO_BN.add(tf.keras.layers.Dropout(0.4))

model_DNN_DO_BN.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model_DNN_DO_BN.compile(
    optimizer= 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_DNN_DO_BN.summary()

# img_input_3 = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# spatial_features_3 = resnet50_model(img_input_3, training=False)
# pooled_features_3 = tf.keras.layers.GlobalAveragePooling2D()(spatial_features_3)

# # --- Dense Block 01 (1024 Nodes) ---
# dense_layer_3_01 = tf.keras.layers.Dense(1024, use_bias=False)(pooled_features_3)
# dense_layer_3_01 = tf.keras.layers.BatchNormalization()(dense_layer_3_01)
# dense_layer_3_01 = tf.keras.layers.LeakyReLU(negative_slope=0.01)(dense_layer_3_01)
# dense_layer_3_01 = tf.keras.layers.Dropout(0.4)(dense_layer_3_01)

# # --- Dense Block 02 (512 Nodes) ---
# dense_layer_3_02 = tf.keras.layers.Dense(512, use_bias=False)(dense_layer_3_01)
# dense_layer_3_02 = tf.keras.layers.BatchNormalization()(dense_layer_3_02)
# dense_layer_3_02 = tf.keras.layers.LeakyReLU(negative_slope=0.01)(dense_layer_3_02)
# dense_layer_3_02 = tf.keras.layers.Dropout(0.4)(dense_layer_3_02)

# # --- Dense Block 03 (256 Nodes) ---
# dense_layer_3_03 = tf.keras.layers.Dense(256, use_bias=False)(dense_layer_3_02)
# dense_layer_3_03 = tf.keras.layers.BatchNormalization()(dense_layer_3_03)
# dense_layer_3_03 = tf.keras.layers.LeakyReLU(negative_slope=0.01)(dense_layer_3_03)
# dense_layer_3_03 = tf.keras.layers.Dropout(0.4)(dense_layer_3_03)

# # --- Final Output Layer ---
# class_probabilities_3 = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(dense_layer_3_03)

# # --- Model Assembly & Compilation ---
# flower_model_v3 = tf.keras.Model(inputs=img_input_3, outputs=class_probabilities_3, name="Robust_BatchNorm_Head")
# flower_model_v3.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )


# In[ ]:


print("--- Training MODEL 4: ResNet & DNN : Batch Normalization + Dropout ---")
t0 = time.time()
resnet_DNN_DO_BN = model_DNN_DO_BN.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
time_taken = time.time() - t0
best_accuracy = max(resnet_DNN_DO_BN.history['val_accuracy']) * 100
print(f"\nBest Val Accuracy: {best_accuracy:.2f}%")
print(f"Training Time:     {time_taken/60:.1f} min")


# In[ ]:


import matplotlib.pyplot as plt

# 1. Set up a clean canvas for two side-by-side plots
plt.figure(figsize=(16, 6))

# --- PLOT 1: Accuracy Comparison (v2 vs v3) ---
plt.subplot(1, 2, 1)
# Model 2: FC Leaky Head / v2 (Green)
plt.plot(resnet_history_FC.history['accuracy'], label='v2 Leaky - Train Acc', color='lightgreen', linestyle='--')
plt.plot(resnet_history_FC.history['val_accuracy'], label='v2 Leaky - Val Acc', color='green')

# Model 3: BatchNorm + Higher Dropout / v3 (Orange)
plt.plot(resnet_history_v3.history['accuracy'], label='v3 BatchNorm - Train Acc', color='moccasin', linestyle='--')
plt.plot(resnet_history_v3.history['val_accuracy'], label='v3 BatchNorm - Val Acc', color='darkorange')

plt.title('Training & Validation Accuracy (v2 vs v3)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)


# --- PLOT 2: Loss Comparison (v2 vs v3) ---
plt.subplot(1, 2, 2)
# Model 2: FC Leaky Head / v2 (Green)
plt.plot(resnet_history_FC.history['loss'], label='v2 Leaky - Train Loss', color='lightgreen', linestyle='--')
plt.plot(resnet_history_FC.history['val_loss'], label='v2 Leaky - Val Loss', color='green')

# Model 3: BatchNorm + Higher Dropout / v3 (Orange)
plt.plot(resnet_history_v3.history['loss'], label='v3 BatchNorm - Train Loss', color='moccasin', linestyle='--')
plt.plot(resnet_history_v3.history['val_loss'], label='v3 BatchNorm - Val Loss', color='darkorange')

plt.title('Training & Validation Loss (v2 vs v3)')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Render the layout cleanly
plt.tight_layout()
plt.show()


# In[ ]:


# --- MODEL 5: ResNet & DNN : Batch Normalization + Dropout + Regularisation ---

resnet50_base.trainable = False
model_DNN_DO_BN_L2 = tf.keras.Sequential()
model_DNN_DO_BN_L2.add(tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

model_DNN_DO_BN_L2.add(resnet50_base)
model_DNN_DO_BN_L2.add(tf.keras.layers.GlobalAveragePooling2D())

model_DNN_DO_BN_L2.add(tf.keras.layers.Dense(1024, use_bias=False, kernel_regularizer=l2(1e-4)))
model_DNN_DO_BN_L2.add(tf.keras.layers.BatchNormalization())
model_DNN_DO_BN_L2.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
model_DNN_DO_BN_L2.add(tf.keras.layers.Dropout(0.4))

model_DNN_DO_BN_L2.add(tf.keras.layers.Dense(512, use_bias=False, kernel_regularizer=l2(1e-4)))
model_DNN_DO_BN_L2.add(tf.keras.layers.BatchNormalization())
model_DNN_DO_BN_L2.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
model_DNN_DO_BN_L2.add(tf.keras.layers.Dropout(0.4))

model_DNN_DO_BN_L2.add(tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=l2( 1e-4)))
model_DNN_DO_BN_L2.add(tf.keras.layers.BatchNormalization())
model_DNN_DO_BN_L2.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
model_DNN_DO_BN_L2.add(tf.keras.layers.Dropout(0.4))

model_DNN_DO_BN_L2.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model_DNN_DO_BN_L2.compile(
    optimizer= 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_DNN_DO_BN_L2.summary()

# img_input_4 = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# spatial_features_4 = resnet50_model(img_input_4, training=False)
# pooled_features_4 = tf.keras.layers.GlobalAveragePooling2D()(spatial_features_4)

# # --- Dense Block 01 (1024 Nodes) ---
# dense_layer_3_01 = tf.keras.layers.Dense(1024, use_bias=False, kernel_regularizer=l2(1e-4))(pooled_features_4)
# dense_layer_3_01 = tf.keras.layers.BatchNormalization()(dense_layer_3_01)
# dense_layer_3_01 = tf.keras.layers.LeakyReLU(negative_slope=0.01)(dense_layer_3_01)
# dense_layer_3_01 = tf.keras.layers.Dropout(0.4)(dense_layer_3_01)

# # --- Dense Block 02 (512 Nodes) ---
# dense_layer_3_02 = tf.keras.layers.Dense(512, use_bias=False,kernel_regularizer=l2(1e-4))(dense_layer_3_01)
# dense_layer_3_02 = tf.keras.layers.BatchNormalization()(dense_layer_3_02)
# dense_layer_3_02 = tf.keras.layers.LeakyReLU(negative_slope=0.01)(dense_layer_3_02)
# dense_layer_3_02 = tf.keras.layers.Dropout(0.4)(dense_layer_3_02)

# # --- Dense Block 03 (256 Nodes) ---
# dense_layer_3_03 = tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=l2(1e-4))(dense_layer_3_02)
# dense_layer_3_03 = tf.keras.layers.BatchNormalization()(dense_layer_3_03)
# dense_layer_3_03 = tf.keras.layers.LeakyReLU(negative_slope=0.01)(dense_layer_3_03)
# dense_layer_3_03 = tf.keras.layers.Dropout(0.4)(dense_layer_3_03)

# # --- Final Output Layer ---
# class_probabilities_3 = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(dense_layer_3_03)

# # --- Model Assembly & Compilation ---
# flower_model_v4 = tf.keras.Model(inputs=img_input_4, outputs=class_probabilities_3, name="Robust_BatchNorm_Head")
# flower_model_v4.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
# flower_model_v4.summary()


# In[ ]:


# print("--- Training Model v4 (Batch Normalization + Dropout + Regularisation) ---")
# t0 = time.time()

# # Save history separately to protect previous model logs
# resnet_history_v4 = flower_model_v4.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# resnet_time_v4 = time.time() - t0
# resnet_best_v4 = max(resnet_history_v4.history['val_accuracy']) * 100
# print(f"\nBest Val Accuracy: {resnet_best_v4:.2f}%")


print("--- Training MODEL 5: ResNet & DNN : Batch Normalization + Dropout + Regularisation ---")
t0 = time.time()
resnet_DNN_DO_BN_L2 = model_DNN_DO_BN_L2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
time_taken = time.time() - t0
best_accuracy = max(resnet_DNN_DO_BN_L2.history['val_accuracy']) * 100
print(f"\nBest Val Accuracy: {best_accuracy:.2f}%")
print(f"Training Time:     {time_taken/60:.1f} min")



# In[ ]:


import matplotlib.pyplot as plt

# 1. Set up a clean canvas for two side-by-side plots
plt.figure(figsize=(16, 6))

# --- PLOT 1: Accuracy (Model v4 Only) ---
plt.subplot(1, 2, 1)
plt.plot(resnet_history_v4.history['accuracy'], label='v4 L2 - Train Acc', color='lightcoral', linestyle='--')
plt.plot(resnet_history_v4.history['val_accuracy'], label='v4 L2 - Val Acc', color='crimson')

plt.title('Model v4: Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)

# --- PLOT 2: Loss (Model v4 Only) ---
plt.subplot(1, 2, 2)
plt.plot(resnet_history_v4.history['loss'], label='v4 L2 - Train Loss', color='lightcoral', linestyle='--')
plt.plot(resnet_history_v4.history['val_loss'], label='v4 L2 - Val Loss', color='crimson')

plt.title('Model v4: Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Render the layout cleanly
plt.tight_layout()
plt.show()


# In[ ]:


# --- MODEL 6: Fine Tunning ResNet & DNN : Batch Normalization + Dropout + Regularisation ---

resnet50_finetune_base = tf.keras.applications.ResNet50(
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False
)

resnet50_finetune_base.trainable = True
total_layers = len(resnet50_finetune_base.layers)

# Refreezing all the layers upto layer 140
for (counter,layer) in enumerate(resnet50_finetune_base.layers[:140]):
    layer.trainable = False
    counter += 1

print(f"Total layers in ResNet-50: {total_layers}")
print(f"Number of trainable layers: {total_layers - counter}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# flower_model_v3.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Tiny learning rate!
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )


# In[ ]:


FINE_TUNE_EPOCHS = 15
TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS

t0 = time.time()

# Notice 'initial_epoch=30' - this keeps history lines continuous!
fine_tune_history = flower_model_v3.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TOTAL_EPOCHS,
    initial_epoch=EPOCHS
)

best_val_acc = max(fine_tune_history.history['val_accuracy']) * 100
print(f"\nFine-Tuning Complete in {(time.time()-t0)/60:.1f} minutes!")
print(f"Best Val Accuracy: {best_val_acc:.2f}%")


# In[ ]:


for layer in resnet50_model.layers:
    if layer.trainable and hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = l2(1e-4)

flower_model_v3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Microscopic steps
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


FINE_TUNE_EPOCHS = 15
TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS

t0 = time.time()

# Notice 'initial_epoch=30' - this keeps history lines continuous!
fine_tune_history_2 = flower_model_v3.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TOTAL_EPOCHS,
    initial_epoch=EPOCHS
)

best_val_acc = max(fine_tune_history.history['val_accuracy']) * 100
print(f"\nFine-Tuning Complete in {(time.time()-t0)/60:.1f} minutes!")
print(f"Best Val Accuracy: {best_val_acc:.2f}%")


# In[ ]:


# FINE_TUNE_EPOCHS = 15
# TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS

# t0 = time.time()

# # Notice 'initial_epoch=30' - this keeps history lines continuous!
# fine_tune_history_2 = flower_model_v3.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=TOTAL_EPOCHS,
#     initial_epoch=EPOCHS
# )

# best_val_acc = max(fine_tune_history.history['val_accuracy']) * 100
# print(f"\nFine-Tuning Complete in {(time.time()-t0)/60:.1f} minutes!")
# print(f"Best Val Accuracy: {best_val_acc:.2f}%")

