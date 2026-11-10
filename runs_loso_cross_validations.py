import os
import random

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from tensorflow.keras.utils import to_categorical
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
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# SETTINGS
# ============================================================

DATASET_FILE = "dataset_mpu_words.csv"

SEQ_LEN = 60
FEATURES = 17

# 60 timesteps × 17 sensor features
TOTAL_FEATURES = SEQ_LEN * FEATURES

REPETITIONS_PER_PARTICIPANT = 20
PARTICIPANTS = 5

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

RESULTS_FILE = "LOSO_Words_results.csv"
SUMMARY_FILE = "LOSO_Words_summary.csv"
CONFUSION_FILE = "LOSO_Words_overall_confusion_matrix.csv"

ASSIGNMENT_FILE = "dataset_mpu_words_LOSO_ready.csv"

MODEL_DIR = "LOSO_Words_models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("LOSO CROSS-VALIDATION FOR CNN-BiLSTM-ATTENTION")
print("=" * 70)


# ============================================================
# LOAD ORIGINAL CSV
# ============================================================

print("\nLoading:", DATASET_FILE)

df = pd.read_csv(
    DATASET_FILE,
    header=None
)

print("Original shape:", df.shape)


# ============================================================
# VERIFY ORIGINAL CSV STRUCTURE
# ============================================================

EXPECTED_COLUMNS = 1 + TOTAL_FEATURES

print("\nExpected columns:")
print("1 label column")
print(f"{TOTAL_FEATURES} sensor columns")
print(f"Total expected columns: {EXPECTED_COLUMNS}")

if df.shape[1] != EXPECTED_COLUMNS:

    raise ValueError(
        f"\nDataset has {df.shape[1]} columns, "
        f"but {EXPECTED_COLUMNS} were expected.\n\n"
        f"Expected:\n"
        f"1 label + {SEQ_LEN} × {FEATURES} sensor values\n"
        f"= {EXPECTED_COLUMNS} columns."
    )


# ============================================================
# IMPORTANT:
# SAVE THE ORIGINAL SENSOR COLUMN REFERENCES
# BEFORE ADDING ANY METADATA
# ============================================================

LABEL_COLUMN_INDEX = 0

SENSOR_COLUMN_INDICES = list(
    range(
        1,
        1 + TOTAL_FEATURES
    )
)

print("\nOriginal data structure verified.")

print("Label column:")
print(f"Column {LABEL_COLUMN_INDEX}")

print(
    f"Sensor columns: "
    f"{SENSOR_COLUMN_INDICES[0]} "
    f"to "
    f"{SENSOR_COLUMN_INDICES[-1]}"
)

print(
    f"Number of sensor columns: "
    f"{len(SENSOR_COLUMN_INDICES)}"
)


# ============================================================
# LABELS
# ============================================================

labels = df.iloc[:, LABEL_COLUMN_INDEX].astype(str)

print("\nNumber of samples:", len(df))
print("Number of classes:", labels.nunique())
print("Classes:", sorted(labels.unique()))


# ============================================================
# CHECK EXPECTED SAMPLE COUNT
# ============================================================

expected_total_samples = (
    PARTICIPANTS
    * labels.nunique()
    * REPETITIONS_PER_PARTICIPANT
)

print(
    "\nExpected total samples:",
    expected_total_samples
)

print(
    "Actual total samples:",
    len(df)
)

if len(df) != expected_total_samples:

    raise ValueError(
        f"\nExpected {expected_total_samples} samples "
        f"but found {len(df)}."
    )


# ============================================================
# RECONSTRUCT PARTICIPANT IDs
# ============================================================
#
# ASSUMPTION:
#
# For every letter:
#
# P01 -> 20 repetitions
# P02 -> 20 repetitions
# P03 -> 20 repetitions
# P04 -> 20 repetitions
# P05 -> 20 repetitions
#
# Then the next letter follows the same structure.
#
# Example:
#
# A: rows 0-19    = P01
# A: rows 20-39   = P02
# A: rows 40-59   = P03
# A: rows 60-79   = P04
# A: rows 80-99   = P05
#
# B: next 100 rows
# etc.
#
# If your CSV was randomized across participants before saving,
# this reconstruction would NOT be valid.
# ============================================================

print("\nReconstructing participant IDs...")

df["label"] = labels

df["repetition_index"] = (
    df.groupby("label")
      .cumcount()
)

df["participant_number"] = (
    df["repetition_index"]
    // REPETITIONS_PER_PARTICIPANT
    + 1
)

df["participant"] = (
    "P"
    + df["participant_number"]
      .astype(str)
      .str.zfill(2)
)


# ============================================================
# VERIFY PARTICIPANT NUMBERS
# ============================================================

unique_participant_numbers = sorted(
    df["participant_number"].unique()
)

expected_participant_numbers = list(
    range(
        1,
        PARTICIPANTS + 1
    )
)

if unique_participant_numbers != expected_participant_numbers:

    raise ValueError(
        "\nParticipant reconstruction produced unexpected "
        f"participant numbers: {unique_participant_numbers}\n"
        f"Expected: {expected_participant_numbers}"
    )


# ============================================================
# VERIFY PARTICIPANT DISTRIBUTION
# ============================================================

print("\nParticipant distribution:")

participant_counts = (
    df["participant"]
      .value_counts()
      .sort_index()
)

print(participant_counts)


expected_per_participant = (
    labels.nunique()
    * REPETITIONS_PER_PARTICIPANT
)

for participant, count in participant_counts.items():

    if count != expected_per_participant:

        raise ValueError(
            f"{participant} has {count} samples; "
            f"expected {expected_per_participant}."
        )


print(
    "\nEach participant has",
    expected_per_participant,
    "samples."
)


# ============================================================
# VERIFY EVERY PARTICIPANT HAS 20 SAMPLES PER CLASS
# ============================================================

class_participant_counts = pd.crosstab(
    df["participant"],
    df["label"]
)

print(
    "\nParticipant × class distribution:"
)

print(class_participant_counts)


if not np.all(
    class_participant_counts.values
    == REPETITIONS_PER_PARTICIPANT
):

    raise ValueError(
        "\nParticipant/class distribution is not exactly "
        f"{REPETITIONS_PER_PARTICIPANT} samples per class."
    )


print(
    "\nVerified: every participant has "
    f"{REPETITIONS_PER_PARTICIPANT} samples "
    "for every gesture class."
)


# ============================================================
# LABEL ENCODING
# ============================================================

encoder = LabelEncoder()

encoder.fit(
    df["label"]
)

num_classes = len(
    encoder.classes_
)

print("\nEncoded classes:")
print(
    list(
        encoder.classes_
    )
)


# ============================================================
# PREPARE SEQUENCES
# ============================================================
#
# IMPORTANT FIX:
#
# DO NOT use:
#
#     row.iloc[2:]
#
# because participant metadata has been added to the dataframe.
#
# Instead, explicitly select the ORIGINAL sensor columns:
#
#     columns 1 through 1020
#
# ============================================================

def prepare_data(data):

    X = []
    y = []

    for _, row in data.iterrows():

        # ----------------------------------------------------
        # SELECT ONLY ORIGINAL SENSOR DATA
        # ----------------------------------------------------

        values = (
            row.iloc[
                SENSOR_COLUMN_INDICES
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        # ----------------------------------------------------
        # VERIFY FEATURE COUNT
        # ----------------------------------------------------

        if len(values) != TOTAL_FEATURES:

            raise ValueError(
                f"Expected {TOTAL_FEATURES} sensor values "
                f"but found {len(values)}."
            )

        # ----------------------------------------------------
        # RESHAPE
        # ----------------------------------------------------

        sequence = values.reshape(
            SEQ_LEN,
            FEATURES
        )

        X.append(
            sequence
        )

        # ----------------------------------------------------
        # ENCODE LABEL
        # ----------------------------------------------------

        label = row["label"]

        encoded_label = encoder.transform(
            [label]
        )[0]

        y.append(
            encoded_label
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    return X, y


# ============================================================
# NORMALIZATION
# ============================================================
#
# IMPORTANT:
#
# Scaling is fitted ONLY on the training participants.
#
# The held-out participant is NEVER used to calculate:
#
#     min
#     max
#
# This prevents data leakage.
# ============================================================

def normalize_using_training_data(
    X_train,
    X_test
):

    X_train = X_train.copy()
    X_test = X_test.copy()

    # --------------------------------------------------------
    # Calculate feature-wise min/max from TRAINING DATA ONLY
    # --------------------------------------------------------

    mins = np.min(
        X_train,
        axis=(0, 1)
    )

    maxs = np.max(
        X_train,
        axis=(0, 1)
    )

    # --------------------------------------------------------
    # Normalize every feature
    # --------------------------------------------------------

    for feature in range(
        FEATURES
    ):

        if maxs[feature] > mins[feature]:

            X_train[:, :, feature] = (
                X_train[:, :, feature]
                - mins[feature]
            ) / (
                maxs[feature]
                - mins[feature]
            )

            X_test[:, :, feature] = (
                X_test[:, :, feature]
                - mins[feature]
            ) / (
                maxs[feature]
                - mins[feature]
            )

        else:

            # Constant feature
            X_train[:, :, feature] = 0.0
            X_test[:, :, feature] = 0.0

    return (
        X_train,
        X_test,
        mins,
        maxs
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    inputs = Input(
        shape=(
            SEQ_LEN,
            FEATURES
        )
    )

    # ========================================================
    # CNN BLOCK 1
    # ========================================================

    x = Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same"
    )(inputs)

    x = MaxPooling1D(
        pool_size=2
    )(x)

    x = Dropout(
        0.3
    )(x)


    # ========================================================
    # CNN BLOCK 2
    # ========================================================

    x = Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same"
    )(x)

    x = MaxPooling1D(
        pool_size=2
    )(x)

    x = Dropout(
        0.3
    )(x)


    # ========================================================
    # BiLSTM 1
    # ========================================================

    x = Bidirectional(
        LSTM(
            128,
            return_sequences=True
        )
    )(x)


    # ========================================================
    # BiLSTM 2
    # ========================================================

    x = Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    )(x)


    # ========================================================
    # SELF ATTENTION
    # ========================================================

    attention = Attention()(
        [x, x]
    )


    # ========================================================
    # RESIDUAL CONNECTION
    # ========================================================

    x = Add()(
        [x, attention]
    )


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    x = GlobalAveragePooling1D()(
        x
    )


    # ========================================================
    # DENSE 1
    # ========================================================

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = BatchNormalization()(
        x
    )

    x = Dropout(
        0.4
    )(x)


    # ========================================================
    # DENSE 2
    # ========================================================

    x = Dense(
        64,
        activation="relu"
    )(x)

    x = BatchNormalization()(
        x
    )

    x = Dropout(
        0.3
    )(x)


    # ========================================================
    # OUTPUT
    # ========================================================

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)


    # ========================================================
    # MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs
    )


    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# PARTICIPANTS
# ============================================================

participants = sorted(
    df["participant"].unique()
)

print("\nParticipants:")
print(
    participants
)


# ============================================================
# STORAGE
# ============================================================

results = []

all_true = []
all_pred = []


# ============================================================
# LOSO CROSS-VALIDATION
# ============================================================

for fold, test_participant in enumerate(
    participants,
    start=1
):

    print("\n\n")

    print(
        "=" * 70
    )

    print(
        f"LOSO FOLD {fold}/{len(participants)}"
    )

    print(
        f"Held-out test participant: "
        f"{test_participant}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # PARTICIPANT-LEVEL SPLIT
    # ========================================================

    train_df = df[
        df["participant"]
        != test_participant
    ].copy()

    test_df = df[
        df["participant"]
        == test_participant
    ].copy()


    print(
        "\nTraining participants:",
        sorted(
            train_df[
                "participant"
            ].unique()
        )
    )

    print(
        "Test participant:",
        test_participant
    )

    print(
        "Training samples:",
        len(train_df)
    )

    print(
        "Test samples:",
        len(test_df)
    )


    # ========================================================
    # PREPARE TRAINING DATA
    # ========================================================

    X_train_all, y_train_all = prepare_data(
        train_df
    )


    # ========================================================
    # PREPARE TEST DATA
    # ========================================================

    X_test, y_test = prepare_data(
        test_df
    )


    # ========================================================
    # VERIFY SHAPES
    # ========================================================

    print(
        "\nTraining X shape:",
        X_train_all.shape
    )

    print(
        "Training y shape:",
        y_train_all.shape
    )

    print(
        "Test X shape:",
        X_test.shape
    )

    print(
        "Test y shape:",
        y_test.shape
    )


    expected_train_samples = (
        expected_per_participant
        * (PARTICIPANTS - 1)
    )

    if len(X_train_all) != expected_train_samples:

        raise ValueError(
            f"Expected {expected_train_samples} "
            f"training samples but got "
            f"{len(X_train_all)}."
        )


    if len(X_test) != expected_per_participant:

        raise ValueError(
            f"Expected {expected_per_participant} "
            f"test samples but got "
            f"{len(X_test)}."
        )


    # ========================================================
    # NORMALIZATION
    # ========================================================
    #
    # TRAINING PARTICIPANTS ONLY
    # ========================================================

    print(
        "\nNormalizing using training "
        "participants only..."
    )

    (
        X_train_all,
        X_test,
        fold_mins,
        fold_maxs
    ) = normalize_using_training_data(
        X_train_all,
        X_test
    )


    # ========================================================
    # SAVE FOLD NORMALIZATION
    # ========================================================

    normalization_path = os.path.join(
        MODEL_DIR,
        f"normalization_{test_participant}.save"
    )

    joblib.dump(
        {
            "mins": fold_mins,
            "maxs": fold_maxs,
            "seq_len": SEQ_LEN,
            "features": FEATURES
        },
        normalization_path
    )

    print(
        "Normalization saved:",
        normalization_path
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================
    #
    # IMPORTANT:
    #
    # Test participant remains completely untouched.
    #
    # Validation comes ONLY from the training participants.
    # ========================================================

    (
        X_train,
        X_val,
        y_train,
        y_val
    ) = train_test_split(

        X_train_all,
        y_train_all,

        test_size=VALIDATION_SIZE,

        random_state=RANDOM_STATE,

        stratify=y_train_all
    )


    print(
        "\nTraining split:",
        X_train.shape
    )

    print(
        "Validation split:",
        X_val.shape
    )


    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    y_train_cat = to_categorical(
        y_train,
        num_classes=num_classes
    )

    y_val_cat = to_categorical(
        y_val,
        num_classes=num_classes
    )


    # ========================================================
    # BUILD FRESH MODEL
    # ========================================================

    print(
        "\nBuilding fresh model..."
    )

    model = build_model()


    # ========================================================
    # PRINT MODEL SUMMARY
    # ========================================================

    if fold == 1:

        print(
            "\nModel architecture:"
        )

        model.summary()


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=15,

        restore_best_weights=True,

        verbose=1
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nTraining fold..."
    )

    history = model.fit(

        X_train,

        y_train_cat,

        validation_data=(
            X_val,
            y_val_cat
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stop
        ],

        verbose=1
    )


    # ========================================================
    # TEST HELD-OUT PARTICIPANT
    # ========================================================

    print(
        "\nTesting:",
        test_participant
    )

    predictions = model.predict(
        X_test,
        verbose=0
    )


    # ========================================================
    # PREDICTED CLASS
    # ========================================================

    y_pred = np.argmax(
        predictions,
        axis=1
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )


    # ========================================================
    # FOLD RESULTS
    # ========================================================

    print(
        "\n------------------------------------------"
    )

    print(
        f"{test_participant} RESULTS"
    )

    print(
        "------------------------------------------"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report"
    )

    report = classification_report(

        y_test,

        y_pred,

        labels=np.arange(
            num_classes
        ),

        target_names=encoder.classes_,

        zero_division=0
    )

    print(
        report
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    fold_cm = confusion_matrix(

        y_test,

        y_pred,

        labels=np.arange(
            num_classes
        )
    )


    print(
        "\nConfusion Matrix"
    )

    print(
        fold_cm
    )


    # ========================================================
    # SAVE FOLD CONFUSION MATRIX
    # ========================================================

    fold_cm_df = pd.DataFrame(

        fold_cm,

        index=encoder.classes_,

        columns=encoder.classes_
    )

    fold_cm_path = os.path.join(

        MODEL_DIR,

        f"confusion_matrix_{test_participant}.csv"
    )

    fold_cm_df.to_csv(
        fold_cm_path
    )

    print(
        "\nFold confusion matrix saved:",
        fold_cm_path
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "fold":
            fold,

        "test_participant":
            test_participant,

        "training_participants":
            ",".join(
                sorted(
                    train_df[
                        "participant"
                    ].unique()
                )
            ),

        "test_samples":
            len(y_test),

        "accuracy":
            accuracy,

        "precision_weighted":
            precision,

        "recall_weighted":
            recall,

        "f1_weighted":
            f1,

        "epochs_trained":
            len(
                history.history[
                    "loss"
                ]
            )
    })


    # ========================================================
    # STORE OUT-OF-FOLD PREDICTIONS
    # ========================================================

    all_true.extend(
        y_test.tolist()
    )

    all_pred.extend(
        y_pred.tolist()
    )


    # ========================================================
    # SAVE FOLD MODEL
    # ========================================================

    model_path = os.path.join(

        MODEL_DIR,

        f"loso_model_{test_participant}.keras"
    )

    model.save(
        model_path
    )

    print(
        "\nFold model saved:",
        model_path
    )


    # ========================================================
    # CLEAN UP
    # ========================================================

    del model
    del history

    tf.keras.backend.clear_session()


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# FINAL LOSO RESULTS
# ============================================================

print("\n\n")

print(
    "=" * 70
)

print(
    "FINAL LOSO RESULTS"
)

print(
    "=" * 70
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# MEAN ± STANDARD DEVIATION
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "MEAN ± STANDARD DEVIATION"
)

print(
    "=" * 70
)


summary_rows = []


for metric in [

    "accuracy",

    "precision_weighted",

    "recall_weighted",

    "f1_weighted"

]:

    mean_value = results_df[
        metric
    ].mean()

    std_value = results_df[
        metric
    ].std(
        ddof=1
    )


    summary_rows.append({

        "metric":
            metric,

        "mean":
            mean_value,

        "std":
            std_value,

        "mean_plus_minus_std":
            f"{mean_value:.4f} ± "
            f"{std_value:.4f}"
    })


    print(
        f"{metric}: "
        f"{mean_value:.4f} ± "
        f"{std_value:.4f}"
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# OVERALL OUT-OF-FOLD CONFUSION MATRIX
# ============================================================
#
# Every prediction here was made on a participant who was
# completely excluded from the model that produced the
# prediction.
# ============================================================

overall_cm = confusion_matrix(

    all_true,

    all_pred,

    labels=np.arange(
        num_classes
    )
)


cm_df = pd.DataFrame(

    overall_cm,

    index=encoder.classes_,

    columns=encoder.classes_
)


cm_df.to_csv(
    CONFUSION_FILE
)


print("\n")

print(
    "=" * 70
)

print(
    "OVERALL LOSO CONFUSION MATRIX"
)

print(
    "=" * 70
)

print(
    cm_df
)


# ============================================================
# OVERALL OUT-OF-FOLD CLASSIFICATION REPORT
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "OVERALL OUT-OF-FOLD CLASSIFICATION REPORT"
)

print(
    "=" * 70
)


overall_report = classification_report(

    all_true,

    all_pred,

    labels=np.arange(
        num_classes
    ),

    target_names=encoder.classes_,

    zero_division=0
)


print(
    overall_report
)


# ============================================================
# OVERALL OUT-OF-FOLD ACCURACY
# ============================================================

overall_accuracy = accuracy_score(
    all_true,
    all_pred
)

print(
    "\nOverall out-of-fold accuracy:",
    f"{overall_accuracy:.4f}"
)


# ============================================================
# SAVE PARTICIPANT-ASSIGNMENT DATASET
# ============================================================
#
# IMPORTANT:
#
# The original sensor columns are copied directly.
#
# The resulting CSV has:
#
# participant
# label
# feature_1
# feature_2
# ...
# feature_1020
# ============================================================

assignment_data = pd.DataFrame()

assignment_data["participant"] = (
    df["participant"]
)

assignment_data["label"] = (
    df["label"]
)


# Copy ONLY the original sensor columns

for i, original_column in enumerate(
    SENSOR_COLUMN_INDICES,
    start=1
):

    assignment_data[
        f"feature_{i}"
    ] = df[
        original_column
    ]


# ============================================================
# VERIFY ASSIGNMENT DATASET
# ============================================================

expected_assignment_columns = (
    2 + TOTAL_FEATURES
)

if (
    assignment_data.shape[1]
    != expected_assignment_columns
):

    raise ValueError(
        "LOSO assignment dataset has "
        f"{assignment_data.shape[1]} columns, "
        f"expected {expected_assignment_columns}."
    )


assignment_data.to_csv(

    ASSIGNMENT_FILE,

    index=False
)


print(
    "\nLOSO-ready dataset saved as:"
)

print(
    ASSIGNMENT_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "LOSO CROSS-VALIDATION COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nFiles generated:"
)

print(
    f"1. {RESULTS_FILE}"
)

print(
    f"2. {SUMMARY_FILE}"
)

print(
    f"3. {CONFUSION_FILE}"
)

print(
    f"4. {ASSIGNMENT_FILE}"
)

print(
    f"5. {MODEL_DIR}/"
)

print(
    "\nFinal overall out-of-fold accuracy:",
    f"{overall_accuracy:.4f}"
)

print(
    "\nEach participant was used exactly once "
    "as an unseen LOSO test participant."
)

print(
    "\n============================================================"
)