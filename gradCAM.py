from torchvision.models import resnet18
from torchcam.cams import GradCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

# Define your model
model = resnet18(pretrained=True).eval()

# Set your CAM extractor
cam_extractor = GradCAM(model)

# Get your input
img = read_image("path/to/your/image.png")
# Preprocess it for your chosen model

# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))

# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map, mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()