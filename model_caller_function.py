import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.ensemble import IsolationForest
import base64
from io import BytesIO
# import torchvision
# import torchvision.transforms as transforms

from tqdm import tqdm
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("KITT.csv")
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')

df = df.set_index("Date")
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

train_size = int(len(df_normalized) * 0.7)
test_size = int(len(df_normalized) * 0.15)
val_size = len(df_normalized) - train_size - test_size

train_data = df_normalized.iloc[:train_size]
test_data = df_normalized.iloc[train_size:train_size + test_size]
val_data = df_normalized.iloc[train_size + test_size:]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx: idx + self.seq_length].values, dtype=torch.float32)
    
seq_length = 10  # Length of input sequences
hidden_size = 64  # Size of the hidden state in the encoder/decoder
latent_dim = 32   # Dimensionality of the latent space
num_layers = 2    # Number of LSTM layers
learning_rate = 1e-3
batch_size = 32
num_epochs = 100

# Create datasets
train_dataset = TimeSeriesDataset(train_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)
val_dataset = TimeSeriesDataset(val_data, seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Assuming input data is normalized between 0 and 1
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    
    def forward(self, x):
        # Encoding
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

# Initialize the VAE
input_dim = train_data.shape[1]  # Number of features in your dataset
latent_dim = 10  # Dimensionality of the latent space
vae = VariationalAutoencoder(input_dim, latent_dim)
print(vae)

def save_and_load_vae(vae_model, path="models", filename="vae_model2.pth"):
  """
  Saves the given VAE model to the specified path and then loads it back.

  Args:
    vae_model: The trained VAE model to be saved and loaded.
    path: The directory where the model will be saved.
    filename: The name of the file to save the model as.

  Returns:
    The loaded VAE model.
  """

  # Save the model
  #torch.save(vae_model.state_dict(), path + "/" + filename)

  # Load the model
  loaded_vae = VariationalAutoencoder(input_dim, latent_dim)  # Assuming these variables are defined in the previous context
  loaded_vae.load_state_dict(torch.load(filename))
  loaded_vae.eval()  # Set the model to evaluation mode

  return loaded_vae

# Example usage:
# Assuming 'vae' is your trained VAE model
vae_model2 = save_and_load_vae(vae)

def generate_and_detect(vae, new_data, seq_length, batch_size, threshold_percentile=95):
  """
  Generates synthetic data, performs anomaly detection on new data, and plots the results.

  Args:
    vae: Trained VariationalAutoencoder model.
    new_data: Pandas DataFrame containing the new time series data for anomaly detection.
    seq_length: Length of input sequences for the VAE.
    batch_size: Batch size for data loading.
    threshold_percentile: Percentile to use for setting the anomaly threshold.

  Returns:
    A tuple containing:
        - Pandas DataFrame with the generated synthetic data
        - Matplotlib figure object for the t-SNE plot
        - Matplotlib figure object for the line plot with anomaly highlights
  """

  # Generate synthetic data (using the same number of samples as new_data)
  #num_samples = len(new_data)
  vae.eval()
  with torch.no_grad():
      z = torch.randn(batch_size, latent_dim)  # Adjust latent_dim if needed
      synthetic_data = vae.decoder(z).numpy()

  #new_data = pd.read_csv(new_data)
  # Convert synthetic data to DataFrame
  print(new_data.head())
  synthetic_df = pd.DataFrame(synthetic_data, columns=new_data.columns)

  # Prepare new data for anomaly detection
  new_dataset = TimeSeriesDataset(new_data, seq_length)  # Assuming TimeSeriesDataset is defined elsewhere
  new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

  # Calculate reconstruction errors on the new dataset
  reconstruction_errors = []
  with torch.no_grad():
    for batch_idx, data in enumerate(new_loader):
        reconstruction, mu, logvar = vae(data)
        batch_errors = F.mse_loss(reconstruction, data, reduction='none').mean(axis=1)
        reconstruction_errors.extend(batch_errors.tolist())

  reconstruction_errors = np.array(reconstruction_errors)

  # Set anomaly threshold
  threshold = np.percentile(reconstruction_errors, threshold_percentile)

  # Identify anomalies in the new dataset
  anomalies = np.where(reconstruction_errors > threshold)[0]
  anomalies = anomalies.tolist()
  # --- t-SNE Plot ---
  # Combine real and synthetic data for t-SNE
  combined_data = np.concatenate((new_data.values, synthetic_data))
  tsne = TSNE(n_components=2, random_state=42)
  embeddings = tsne.fit_transform(combined_data)
  real_embeddings = embeddings[:len(new_data)]
  synthetic_embeddings = embeddings[len(new_data):]

  # Create t-SNE figure
  tsne_fig = plt.figure(figsize=(8, 6))
  plt.scatter(real_embeddings[:, 0], real_embeddings[:, 1], label='Real Data', alpha=0.7)
  plt.scatter(synthetic_embeddings[:, 0], synthetic_embeddings[:, 1], label='Synthetic Data', alpha=0.7)
  plt.title('t-SNE Visualization of Real and Synthetic Data')
  plt.xlabel('t-SNE Dimension 1')
  plt.ylabel('t-SNE Dimension 2')
  plt.legend()

  img_buffer = BytesIO()
  plt.savefig(img_buffer, format='png')
  img_buffer.seek(0)

    # Convert image to base64 string
  #img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
  plt.close(tsne_fig)
  # --- Line Plot with Anomalies ---
  # Create line plot figure
  
  #line_fig = plt.figure(figsize=(15, 6))
  #for i in range(new_data.shape[1]):
     # plt.plot(new_data.index, new_data.iloc[:, i], label=new_data.columns[i])

  # Highlight anomalies
  #for idx in anomalies:
  #    start_idx = idx
  #    end_idx = start_idx + seq_length
  #    plt.axvspan(new_data.index[start_idx], new_data.index[end_idx - 1], color='red', alpha=0.3)

  #plt.title("Anomalies Detected in New Time Series Data")
  #plt.xlabel("Time")
  #plt.ylabel("Value")
  #plt.legend()

  return synthetic_df,anomalies,img_buffer,#tsne_fig, line_fig

# Example usage:
# Assuming 'vae', 'new_df', 'seq_length', and 'latent_dim' are defined

#synthetic_df, anomalies = generate_and_detect(vae, df, seq_length, batch_size=len(df))

#print("Synthetic Data:")
#print(synthetic_df.head())
#print(anomalies)

# Display the plots
#tsne_fig.show()
#line_fig.show()
