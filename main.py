import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from torch.utils.data import DataLoader

from preprocessing import preprocess_data
from lstm-decoder import EEGDataset, EEG_LSTM, train_model

# ----------------------
# Step 1: Preprocess EEG data
# ----------------------
labels, data, info = preprocess_data(nb_subjects=5, run_type=2, band='mu', crop=(1.0, 2.0))
X = data['mu']  # shape (N, C, T)
y = labels

# ----------------------
# Step 2: Train/test split
# ----------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Dummy classifier (chance-level benchmark)
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train.reshape(len(X_train), -1), y_train)
chance_acc = dummy.score(X_val.reshape(len(X_val), -1), y_val) * 100
print(f"Chance Accuracy (Dummy): {chance_acc:.2f}%")

# ----------------------
# Step 3: Train LSTM
# ----------------------
train_dataset = EEGDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = EEG_LSTM(input_size=X.shape[1], num_classes=2)
model, train_acc_list, val_acc_list = train_model(model, train_loader, (X_val, y_val), epochs=30)

# ----------------------
# Step 4: Plot Accuracy
# ----------------------
plt.figure(figsize=(10, 6))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.hlines(chance_acc, 0, len(val_acc_list)-1, colors='red', linestyles='dashed', label='Chance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('LSTM vs Chance on Motor Imagery Decoding')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
