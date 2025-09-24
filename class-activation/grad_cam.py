
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as functional
from torchvision import models, transforms

import warnings
warnings.filterwarnings('ignore')

# Load pretrained model
model = models.resnet18(weights=True)
print(model)
print("\n\n")

model.eval()

# Choose the target layer
target_layer = model.layer4[1].conv2  # Final conv layer in ResNet18

# Load and preprocess image
img_path = './pizza_steak_sushi/test/pizza/648055.jpg'
img = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
                                ])
input_tensor = transform(img).unsqueeze(0)

# Hook to capture activations and gradients
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])


# Register hooks
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Forward pass
logits = model(input_tensor)
output = logits
print("----")
print(logits.shape)
pred_class = output.argmax(dim=1)
print(pred_class)
print(output[0, pred_class])

# Backward pass
model.zero_grad()
# compute gradients of most confident logit with respect to all model's params 
output[0, pred_class].backward()
print()

# Get activations and gradients
act = activations[0].squeeze().detach()
grad = gradients[0].squeeze().detach()
print(act.shape, '\n', grad.shape)

# Compute weights and Grad-CAM
weights = grad.mean(dim=(1, 2))  # Global average pooling
cam = torch.zeros(act.shape[1:], dtype=torch.float32)
print(weights.shape)
print(cam.shape)

# multiply each activation map channel by its corresponding weight and sum them up
# 'act[i]' shows what spatial features that channel responded to
# 'w' represents how much that channel influences output.
for i, w in enumerate(weights):
    cam += w * act[i]

cam = functional.relu(cam)  # Apply ReLU
cam = cam - cam.min()
cam = cam / cam.max()

# resulting cam shows where the model was “looking” when it made its prediction
cam = cam.numpy()

# Resize and overlay on image
heatmap = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
img_np = np.array(img.resize((224, 224)))
overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

# Show result
cv2.imshow("Grad-CAM Overlay", overlay)
cv2.waitKey(0)

cv2.destroyAllWindows()



