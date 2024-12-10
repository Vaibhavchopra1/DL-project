import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class RealFakeClassifier(nn.Module):
    def __init__(self):
        super(RealFakeClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25), 
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25) 
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

# Instantiate the model
model = RealFakeClassifier()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation(10),  # Augmentation: Random Rotation
    transforms.RandomHorizontalFlip(),  # Augmentation: Horizontal Flip
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root='/home/mangesh_singh/testing_project/currency', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(root='/home/mangesh_singh/testing_project/currency/validation', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_epochs = 20
best_val_accuracy = 0  


for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Training Phase
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)
    
    train_accuracy = total_correct / total_samples
    
    # Validation Phase
    model.eval()
    val_correct = 0
    val_samples = 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_correct += (val_outputs.argmax(1) == val_labels).sum().item()
            val_samples += val_labels.size(0)
    
    val_accuracy = val_correct / val_samples * 100  # Convert to percentage

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'real_fake_classifier_best.pth')

    # Print Epoch Summary
    print(f"{epoch:<10}{total_loss / len(train_loader):<15.4f}{val_accuracy:<25.2f}")

print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")