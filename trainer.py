import torch
import torch.nn as nn
from torchvision import datasets
from model import NeuralNetwork, TRANSFORM
import matplotlib.pyplot as plt
import os

# Training parameters
EPOCHS = 7
BATCH_SIZE = 100
LEARNING_RATE = .001

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training MNIST training dataset
train_minst = datasets.MNIST("./data", train=True, download=True, transform=TRANSFORM)
test_minst = datasets.MNIST("./data", train=False, download=True, transform=TRANSFORM)

# Data loaders for batching
DataLoader = torch.utils.data.DataLoader
train_loader = DataLoader(train_minst, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_minst, batch_size=BATCH_SIZE)

# Instantiate NN model & set device to CUDA
model = NeuralNetwork().to(device)
model_path = "model.pt"

# Loss function & optimzer
loss = nn.CrossEntropyLoss()
optimzer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load model if exists
if os.path.exists(model_path):
	model.load_state_dict(torch.load(model_path))
	model.eval()

# Begin training
else:
	for epoch in range(EPOCHS):
		for i, (images, labels) in enumerate(train_loader):

			# Prepare input for training
			images = images.to(device).reshape(-1, 784)
			labels = labels.to(device)

			# Forward pass
			outputs = model(images)
			loss_value = loss(outputs, labels)

			# Backpropogation
			optimzer.zero_grad()
			loss_value.backward()
			optimzer.step()

			if (i+1) % 100 == 0:
				print(f"Epoch [{epoch + 1}/{EPOCHS}]; Step [{i + 1}/{len(train_loader)}]; Loss: {loss_value.item():.4f};")

	# Save model after training
	torch.save(model.state_dict(), "model.pt")

# Test data
with torch.no_grad():
	n_samples = 0
	n_correct = 0
	for images, labels in test_loader:
		images = images.to(device).reshape(-1, 784)
		labels = labels.to(device)
		outputs = model(images)

		# max returns (value, index)
		_, predicted = torch.max(outputs.data, 1)
		n_samples += labels.size(0)
		n_correct += (predicted == labels).sum().item()

	acc = 100.0 * n_correct / n_samples
	print(f'Accuracy of the network on the 10000 test images: {acc:.2f} %')
	
img, label = test_minst[0]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.savefig("img.png")