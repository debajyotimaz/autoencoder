import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Define and parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--val_split', type=float, default=0.2, help='fraction of input data to use for validation')
args = parser.parse_args()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
25
class SentimentAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SentimentAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, output_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define hyperparameters
input_size = 10000 # Size of input tensor
hidden_size = 200 # Size of hidden layer
output_size = 25 # Size of output tensor
lr = 0.001 # Learning rate
epochs = args.epochs # Number of epochs (from command-line argument)
batch_size = args.batch_size # Batch size (from command-line argument)
validation_split = args.val_split # Fraction of input data to use for validation (from command-line argument)
25
# Instantiate the model and move to GPU
model = SentimentAutoencoder(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Generate some fake input data and move to GPU
input_data = torch.rand(10000, input_size).to(device)

# Split input data into training and validation sets
n_validation = int(validation_split * input_data.size(0))
n_training = input_data.size(0) - n_validation
training_data, validation_data = random_split(input_data, [n_training, n_validation])
print("train size:",n_training)
print("n_validation:",n_validation)

# Create data loaders for training and validation sets
training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

# Lists to store loss and epoch values
train_loss = []
val_loss = []
epoch_list = []

# Train the model
for epoch in range(epochs):
    running_loss = 0.0
    running_val_loss = 0.0

    # Training loop
    for inputs in training_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0) / n_training

    # Validation loop
    with torch.no_grad():
        for inputs in validation_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            running_val_loss += loss.item() * inputs.size(0) / n_validation

    # Calculate average losses for the epoch
    epoch_loss = running_loss
    epoch_val_loss = running_val_loss

    # Append to lists
    train_loss.append(epoch_loss)
    val_loss.append(epoch_val_loss)
    epoch_list.append(epoch+1)

    # Print progress
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

## Extract the encoder output and move to CPU
#encoder_output = model.encoder(input_data).cpu()
#print(encoder_output.shape)

# Plot the loss vs epoch graph
plt.plot(epoch_list, train_loss, label='Training Loss')
plt.plot(epoch_list, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Sentiment Autoencoder Loss')
plt.legend()
plt.show()


