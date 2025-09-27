from torchvision import transforms
import torch.nn as nn

INPUTS = 784
HIDDEN_NODES0 = 128
OUTPUTS = 10

TRANSFORM = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((.5, ), (.5, )),
])

# Neural network model
class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		
		# Network layers
		self.inputs = nn.Linear(INPUTS, HIDDEN_NODES0)
		self.relu = nn.ReLU()
		self.outputs = nn.Linear(HIDDEN_NODES0, OUTPUTS)

	# Forward pass method
	def forward(self, x):
		out = self.inputs(x)
		out = self.relu(out)
		out = self.outputs(out)
		return out