import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ------------------------
# Step 1: Load CSV
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="Path to input CSV file")
args = parser.parse_args()

df = pd.read_csv(args.csv)

# ------------------------
# Step 2: Clean column names
# ------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.replace("Â", "")
    .str.replace("°", "c")
    .str.lower()
    .str.replace(r"c{2,}", "c", regex=True)
)

print("\n✅ Cleaned Columns:", df.columns.tolist())
print(df.head())

# ------------------------
# Step 3: Generate health labels (softened for more uniform distribution)
# ------------------------
def assign_health(row):
    ph = row["ph"]
    sal = row["salinity (psu)"]
    turb = row["turbidity (ntu)"]
    temp = row["temperature (c)"]

    # Critical: extreme values
    if (ph < 6.9 or ph > 8.3) or (sal < 31 or sal > 37) or (turb > 6) or (temp < 20 or temp > 29):
        return "Critical"
    
    # Healthy: ideal range
    elif (7.5 <= ph <= 8.0) and (33 <= sal <= 35) and (turb <= 2) and (24 <= temp <= 27):
        return "Healthy"
    
    # Moderate: everything else
    else:
        return "Moderate"

df["health_label"] = df.apply(assign_health, axis=1)

print("\n✅ Sample with generated labels:")
print(df.head())

# ------------------------
# Step 4: Prepare data
# ------------------------
X = df[["ph", "salinity (psu)", "turbidity (ntu)", "temperature (c)"]].values
y = df["health_label"].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------
# Step 5: Train-test split
# ------------------------
X_train_full, X_test, y_train_full_encoded, y_test_encoded = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert to categorical AFTER splitting
y_train_full = to_categorical(y_train_full_encoded, num_classes=num_classes)
y_test = to_categorical(y_test_encoded, num_classes=num_classes)

# Split train into train + validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full_encoded
)

print("Training set class counts:", np.bincount(y_train_full_encoded))
print("Test set class counts:", np.bincount(y_test_encoded))

# ------------------------
# Step 6: Define FNN model
# ------------------------
model = Sequential([
    Dense(128, input_shape=(4,), activation="relu"),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax")  # dynamic output size
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# ------------------------
# Step 7: Train model
# ------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# ------------------------
# Step 8: Evaluate model
# ------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# Confusion Matrix and Classification Report
y_pred_classes = np.argmax(model.predict(X_test), axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nConfusion Matrix (3x3):")
print(confusion_matrix(y_true_classes, y_pred_classes, labels=range(num_classes)))

print("\nClassification Report:")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    labels=range(num_classes),
    target_names=encoder.classes_
))

# ------------------------
# Step 9: Save model
# ------------------------
model.save("ecosystem_health_fnn.keras")  # native Keras format
print("\n💾 Model saved as ecosystem_health_fnn.keras")

# ------------------------
# Optional: Plot training history
# ------------------------
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('FNN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('FNN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("training_accuracy.png")  # saves the figure
print("\n✅ Training accuracy plot saved as training_accuracy.png")



# ------------------------
# Step 10: Visualizations
# ------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure 'timestamp' column is datetime
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

features = ["ph", "salinity (psu)", "turbidity (ntu)", "temperature (c)"]
target = "health_label"

# 1️⃣ Box plots for all features
plt.figure(figsize=(12,6))
sns.boxplot(data=df[features])
plt.title("Box Plot of Features")
plt.savefig("box_plots.png")
plt.show()

# 2️⃣ Histograms + KDE for each feature
for feat in features:
    plt.figure(figsize=(8,4))
    sns.histplot(df[feat], kde=True, bins=30)
    plt.title(f"Histogram + KDE for {feat}")
    plt.savefig(f"hist_{feat}.png")
    plt.show()

# 3️⃣ Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Features")
plt.savefig("correlation_heatmap.png")
plt.show()

# 4️⃣ Pairplot colored by health_label
sns.pairplot(df[features + [target]], hue=target, diag_kind='kde', corner=True)
plt.savefig("pairplot_health.png")
plt.show()

# 5️⃣ Time trend plots (if timestamp available)
if 'timestamp' in df.columns:
    for feat in features:
        plt.figure(figsize=(10,4))
        sns.lineplot(data=df, x='timestamp', y=feat, hue=target)
        plt.title(f"Time Trend of {feat}")
        plt.xlabel("Timestamp")
        plt.ylabel(feat)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"time_trend_{feat}.png")
        plt.show()

# 6️⃣ Bar plot for class distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x=target, order=df[target].value_counts().index)
plt.title("Distribution of Health Labels")
plt.savefig("class_distribution.png")
plt.show()
