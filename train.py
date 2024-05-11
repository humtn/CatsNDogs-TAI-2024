import sys
import pandas as pd
import shutil
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from PIL import Image 

# Load datasets (1000 features)
train_features = torch.load('features/train_features.pth')
train_labels = torch.load('features/train_labels.pth')
test_features = torch.load('features/test_features.pth')
test_labels = torch.load('features/test_labels.pth')

train_labels = train_labels.float()
test_labels = test_labels.float()

input_size = 1000
output_size = 1

final_hidden_neurons = 512
final_dropout_rate = 0.3
final_learning_rate = 0.001
final_weight_decay = 0.000001
final_num_epochs = 80

class cats_vs_dogs(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(cats_vs_dogs, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = cats_vs_dogs(input_size, hidden_size=final_hidden_neurons, output_size=1, dropout=final_dropout_rate)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=final_learning_rate, weight_decay=final_weight_decay)

for epoch in range(79):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_labels.unsqueeze(1))  # BCELoss expects 2D output
    loss.backward()
    optimizer.step()

    # Evaluate each epoch for early stopping
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_features)
        test_loss = criterion(test_outputs, test_labels.unsqueeze(1))
        test_predictions = (test_outputs >= 0.5).float()
        test_accuracy = (test_predictions == test_labels.unsqueeze(1)).float().mean().item()
        
# Print the accuracy and loss at the final epoch
print(f"Final Test Loss: {test_loss.item()}, Final Test Accuracy: {test_accuracy}")

# Export final model
torch.save(model.state_dict(), 'final_model')