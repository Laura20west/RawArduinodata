import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
       Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling1D,
    Attention,
    Add
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ==========================================
# SETTINGS
# ==========================================

DATASET_FILE = "dataset_mpuallalpha.csv"

SEQ_LEN = 60
FEATURES = 17

# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_FILE,
    header=None
)

labels = df.iloc[:, 0].values

raw_data = df.iloc[:, 1:].values

print("Rows:", len(df))

# ==========================================
# REBUILD SEQUENCES
# ==========================================

expected_length = (
    SEQ_LEN * FEATURES
)

X = []
y = []

for i in range(len(raw_data)):

    row = raw_data[i]

    if len(row) < expected_length:
        continue

    row = row[:expected_length]

    try:

        gesture = np.array(
            row,
            dtype=np.float32
        ).reshape(
            SEQ_LEN,
            FEATURES
        )

        X.append(gesture)

        y.append(labels[i])

    except Exception:
        continue

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)

print("\nProcessed Data")

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ==========================================
# LABEL ENCODING
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# ==========================================
# REMOVE CLASSES WITH <2 SAMPLES
# ==========================================

unique, counts = np.unique(
    y_encoded,
    return_counts=True
)

valid_classes = unique[
    counts >= 2
]

mask = np.isin(
    y_encoded,
    valid_classes
)

X = X[mask]
y = y[mask]

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_cat = to_categorical(
    y_encoded
)

print(
    "\nClasses:",
    len(encoder.classes_)
)

print(
    "Samples:",
    len(X)
)

# ==========================================
# FEATURE NORMALIZATION
# ==========================================

print("\nNormalizing...")

mins = []
maxs = []

for feature in range(FEATURES):

    feature_data = X[:, :, feature]

    min_val = feature_data.min()
    max_val = feature_data.max()

    mins.append(float(min_val))
    maxs.append(float(max_val))

    if max_val > min_val:

        X[:, :, feature] = (
            feature_data - min_val
        ) / (
            max_val - min_val
        )

joblib.dump(
    {
        "mins": mins,
        "maxs": maxs
    },
    "normalization.save"
)

print(
    "Normalization saved."
)

# ==========================================
# 70:15:15 SPLIT
# ==========================================

print(
    "\nCreating Train/Val/Test Split..."
)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_cat,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

temp_labels = np.argmax(
    y_temp,
    axis=1
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

print(
    "\nTrain:",
    len(X_train)
)

print(
    "Validation:",
    len(X_val)
)

print(
    "Test:",
    len(X_test)
)

# ==========================================
# MODEL
# ==========================================

print("\nBuilding CNN-BiLSTM-Attention...")

inputs = Input(shape=(SEQ_LEN, FEATURES))

# ==========================================
# Conv Block 1
# ==========================================

x = Conv1D(
    filters=64,
    kernel_size=3,
    activation="relu",
    padding="same"
)(inputs)

x = MaxPooling1D(pool_size=2)(x)

x = Dropout(0.3)(x)

# ==========================================
# Conv Block 2
# ==========================================

x = Conv1D(
    filters=128,
    kernel_size=3,
    activation="relu",
    padding="same"
)(x)

x = MaxPooling1D(pool_size=2)(x)

x = Dropout(0.3)(x)

# ==========================================
# BiLSTM Layer 1
# ==========================================

x = Bidirectional(
    LSTM(
        128,
        return_sequences=True
    )
)(x)

# ==========================================
# BiLSTM Layer 2
# ==========================================

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(x)

# ==========================================
# Self Attention
# ==========================================

attention = Attention()([x, x])

# Residual Connection
x = Add()([x, attention])

# ==========================================
# Pooling
# ==========================================

x = GlobalAveragePooling1D()(x)

# ==========================================
# Dense Layer
# ==========================================

x = Dense(
    128,
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Dropout(0.4)(x)

# ==========================================
# Dense Layer
# ==========================================

x = Dense(
    64,
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Dropout(0.3)(x)

# ==========================================
# Output Layer
# ==========================================

outputs = Dense(
    len(encoder.classes_),
    activation="softmax"
)(x)

model = Model(
    inputs=inputs,
    outputs=outputs
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary\n")

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_allalpha_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ==========================================
# TRAIN
# ==========================================

print("\nTraining...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=100,

    batch_size=32,

    callbacks=[
        early_stop,
        checkpoint
    ]
)

# ==========================================
# EVALUATE
# ==========================================

print(
    "\nEvaluating..."
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    "\nTest Accuracy:",
    accuracy
)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

predictions = model.predict(
    X_test
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)

print(
    "\nClassification Report"
)

print(

    classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

print(
    "\nConfusion Matrix"
)

print(

    confusion_matrix(
        y_true,
        y_pred
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

model.save(
    "glove_model_mpuallalpha.keras"
)

# ==========================================
# SAVE LABELS
# ==========================================

with open(
    "labels.txt",
    "w"
) as f:

    for label in encoder.classes_:

        f.write(
            label + "\n"
        )

print(
    "\nTraining Complete"
)

print(
    "Model Saved"
)

print(
    "Labels Saved"
)

print(
    "Normalization Saved"
)