from ultralytics import YOLO
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
import torch

# Initialize YOLOv8 model
model = YOLO("yolov8l.pt")  # Use "yolov8l-seg.pt" for masks (recommended)

# Detect objects
results = model.predict(
    source="C:/Users/ual-laptop/AI&XRStudio/ObjectDetection/results/table.png",
    show=True,
    save=True,
    show_labels=True,
    iou=0.45
)

# Load Stable Diffusion for texture generation
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cpu")  # Requires NVIDIA GPU with ≥8GB VRAM

# Read original image
original_img = cv2.imread("C:/Users/ual-laptop/AI&XRStudio/ObjectDetection/results/table.png")

# Process each detected object
# Inside your object detection loop
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Calculate original dimensions
        original_height = y2 - y1
        original_width = x2 - x1
        
        # Adjust dimensions to be divisible by 8
        adjusted_height = (original_height // 8) * 8  # Round down
        adjusted_width = (original_width // 8) * 8    # Round down
        
        # Ensure minimum size (e.g., 64x64)
        adjusted_height = max(adjusted_height, 64)
        adjusted_width = max(adjusted_width, 64)
        
        # Generate texture with adjusted dimensions
        texture = pipe(
            prompt="realistic texture", 
            height=adjusted_height, 
            width=adjusted_width
        ).images[0]
        
        # Resize texture to original dimensions
        texture_resized = texture.resize((original_width, original_height))
        
        # Convert to OpenCV format and blend
        texture_cv = cv2.cvtColor(np.array(texture_resized), cv2.COLOR_RGB2BGR)
        mask = 255 * np.ones((original_height, original_width), dtype=np.uint8)
        
        # Apply to original image
        blended = cv2.seamlessClone(
            texture_cv, 
            original_img[y1:y2, x1:x2], 
            mask, 
            (x1, y1), 
            cv2.NORMAL_CLONE
        )
        original_img[y1:y2, x1:x2] = blended

# Save and show final result
cv2.imwrite("textured_object.jpg", original_img)
cv2.imshow("Textured Result", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()