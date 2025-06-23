import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# 1. Binning Function
def bin_wellbeing(productivity, stress):
    score = productivity - (stress * 5)
    return pd.cut(score, bins=[-float('inf'), 40, 70, float('inf')], labels=[0, 1, 2]).astype(int)

# 2. Data Loader
def load_data_from_csv(path='mental_health_data.csv'):
    df = pd.read_csv(path)
    y_class = bin_wellbeing(df['productivity_score'], df['stress_level'])

    if y_class.isnull().any():
        df = df[~y_class.isnull()]
        y_class = bin_wellbeing(df['productivity_score'], df['stress_level'])

    X = df.drop(columns=['productivity_score']).values
    y = y_class.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
        scaler
    )

# 3. Dataset Class
class MentalHealthDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4. Model Architecture
def build_model(input_size=4, num_classes=3):
    return nn.Sequential(
        nn.Linear(input_size, 16),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, num_classes)
    )

# 5. Training Loop
def train_model(model, dataloader, val_loader=None, epochs=15, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if val_loader:
            acc = evaluate_model(model, val_loader)
            print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {acc:.2%}")

# 6. Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total

# 7. Load New User Data from File
def load_new_user_data(file_path='new_user_input.txt'):
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        values = [float(x) for x in line.split(',')]
    return np.array([values])

# 8. New User Prediction
def predict_new_user(model_path, scaler, input_size=4, num_classes=3, file_path='new_user_input.txt'):
    model = build_model(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    new_input_array = load_new_user_data(file_path)
    new_scaled = scaler.transform(new_input_array)
    new_tensor = torch.tensor(new_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(new_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

# 9. Save Model
def save_model(model, path='mental_model_class.pth'):
    torch.save(model.state_dict(), path)

# 10. Main Execution
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler = load_data_from_csv('mental_health_data.csv')

    train_dataset = MentalHealthDataset(X_train, y_train)
    test_dataset = MentalHealthDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    model = build_model(input_size=X_train.shape[1])
    train_model(model, train_loader, val_loader=test_loader, epochs=15)

    accuracy = evaluate_model(model, test_loader)
    print(f" Final Test Accuracy: {accuracy:.2%}")

    save_model(model)
    print(" Model saved to 'mental_model_class.pth'")

    # Predict new user data from file
    predicted = predict_new_user('mental_model_class.pth', scaler, file_path='new_user_input.txt')
    print(f" Predicted Mental Wellbeing Class: {predicted}")
