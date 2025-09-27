import torch
from model import NeuralNetwork, TRANSFORM
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

# Set device if GPU exists
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate NN model
model = NeuralNetwork().to(device)
model_path = "model.pt"

# Load model weights otherwise exist
if not os.path.exists(model_path):
	print(f"No model \"{model_path}\" found in directory!")
	exit()
model.load_state_dict(torch.load(model_path))
model.eval()

# GUI
class Window():
	def __init__(self, root: tk.Tk):
		self.root = root
		root.title("Digit Classifier")
		root.resizable(False, False)

		# Initilize grid size
		self.canvas_size = 280
		self.grid_size = 28
		self.pixel_size = self.canvas_size / self.grid_size

		# PIL Image
		self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
		self.draw = ImageDraw.Draw(self.image)
		self.tk_image = ImageTk.PhotoImage(self.image)

		# Canvas setup
		self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, cursor="cross", bg="black")
		self.canvas.grid()
		self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
		self.canvas.bind("<B1-Motion>", self.paint)
		self.canvas.bind("<ButtonRelease-1>", self.predict)

		# Clear button
		self.clear_button = tk.Button(root, text="CLEAR", command=self.clear_canvas, height=2)
		self.clear_button.grid(sticky="ew")

	# Process image and output results
	def predict(self, _):

		# Resize image to 28x28
		resized = self.image.resize((28, 28))
		
		# resized.save("a.png")

		# Convert image to tensor of floats
		img_tensor = TRANSFORM(resized).float()

		# Convert tensor to match model input (test data)
		img_tensor = img_tensor.squeeze(0).to(device)

		# Calculate output
		out = ""
		with torch.no_grad():
			outputs = model(img_tensor.reshape(-1, 784))

			probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
			for i, probability in enumerate(probabilities):
				out += f"{i}: {probability * 100:.0f}%\n"

			_, predicted = torch.max(outputs.data, 1)
		
		print(f"Predicted: [{predicted.item()}]\n{out}")

	# Primary paint method that fires at every drag event
	def paint(self, e):
		x1, y1 = (e.x - 10), (e.y - 10)
		x2, y2 = (e.x + 10), (e.y + 10)

		# Draw on PIl image
		self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")
		self.paste_image()
	
	# Clears canvas on press
	def clear_canvas(self):
		self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill="black")
		self.paste_image()

	# Paste PIL image to canvas
	def paste_image(self):
		self.tk_image.paste(self.image.resize((28, 28)).resize((280, 280), Image.Resampling.BOX))

# Run program
root = tk.Tk()
Window(root)
root.mainloop()

