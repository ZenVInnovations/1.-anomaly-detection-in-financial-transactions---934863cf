import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
from tqdm import tqdm

class EnhancedAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, input_dim)  # No ReLU/Sigmoid on output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_encoded(self, x):
        return self.encoder(x)

def normalize_data(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1, std)
    return (X - mean) / std, mean, std

def run_autoencoder(X, true_labels=None, threshold=None, epochs=50, learning_rate=0.001, batch_size=32):
    X_original = X.copy()

    if X.isna().any().any():
        X = X.fillna(X.mean())

    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    X_norm, mean, std = normalize_data(X_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedAutoencoder(input_dim=X_norm.shape[1]).to(device)
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        reconstruction_errors = torch.mean(criterion(reconstructed, X_tensor), dim=1).cpu().numpy()

    if threshold is None:
        threshold = np.percentile(reconstruction_errors, 98)

    anomaly_scores = (reconstruction_errors > threshold).astype(int)

    metrics = None
    if true_labels is not None:
        precision = precision_score(true_labels, anomaly_scores, zero_division=0)
        recall = recall_score(true_labels, anomaly_scores, zero_division=0)
        f1 = f1_score(true_labels, anomaly_scores, zero_division=0)
        accuracy = accuracy_score(true_labels, anomaly_scores)
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "threshold": threshold,
            "reconstruction_errors": reconstruction_errors.tolist()
        }

    return X_original, anomaly_scores, metrics
