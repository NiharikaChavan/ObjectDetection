from ultralytics import YOLO
import cv2
import numpy as np
from diffusers import StableDiffusionXLPipeline
import torch
import gc

# Check for GPU availability and memory
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

try:
    # Initialize YOLOv8 segmentation model
    model = YOLO("yolov8m-seg.pt").to(device)  # Using medium segmentation model for better accuracy
    
    print("Running object detection and segmentation...")
    # Detect objects
    results = model.predict(
        source="C:/Users/AICoreXR/TextureSynthesis/ObjectDetection/results/table.png",
        show=False,  # Don't show the visualization
        save=False,  # Don't save the visualization
        show_labels=True,
        iou=0.45,
        conf=0.25  # Lower confidence threshold to catch more parts
    )

    print("Loading Stable Diffusion pipeline...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)

        pipe.enable_attention_slicing(slice_size="auto")
        
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        print("Reading original image...")
        original_img = cv2.imread("C:/Users/AICoreXR/TextureSynthesis/ObjectDetection/results/table.png")
        if original_img is None:
            raise ValueError("Failed to read the image file")

        # Get image dimensions
        height, width = original_img.shape[:2]
        
        # Create a mask for table top (upper part) and legs (lower part)
        table_top_mask = np.zeros((height, width), dtype=np.uint8)
        table_legs_mask = np.zeros((height, width), dtype=np.uint8)

        print("Processing detected objects...")
        for result in results:
            if result.masks is not None:
                for seg, box, cls in zip(result.masks.data, result.boxes, result.boxes.cls):
                    # Get class name
                    class_name = result.names[int(cls)]
                    
                    # Skip if not a table
                    if class_name.lower() != 'dining table':
                        print(f"Skipping non-table object: {class_name}")
                        continue

                    # Convert segment to numpy array
                    segment = seg.cpu().numpy()
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    box_height = y2 - y1
                    
                    # Create binary mask from segment
                    binary_mask = (segment > 0.5).astype(np.uint8) * 255
                    
                    # Split into top and bottom parts
                    mid_y = y1 + int(box_height * 0.3)  # Adjust split point (30% from top)
                    
                    # Assign to respective masks
                    table_top_mask[y1:mid_y, x1:x2] = binary_mask[y1-y1:mid_y-y1, x1-x1:x2-x1]
                    table_legs_mask[mid_y:y2, x1:x2] = binary_mask[mid_y-y1:y2-y1, x1-x1:x2-x1]

        # Process table top
        if np.any(table_top_mask):
            prompt = "ultra detailed photograph of polished aluminum table surface, mirror-like metallic finish, brushed steel texture, cold metal surface with light reflections, industrial metalwork, raw metal material, professional studio lighting, 8k uhd"
            try:
                with torch.autocast(device):
                    # Generate multiple textures and blend them
                    textures = []
                    for _ in range(2):  # Generate 2 variations
                        texture = pipe(
                            prompt=prompt,
                            negative_prompt="wood, wooden, grainy, organic, natural, warm colors, brown, beige, painted, plastic",
                            height=768,
                            width=768,
                            guidance_scale=15.0,  # Increased for more pronounced effect
                            num_inference_steps=50
                        ).images[0]
                        texture_cv = cv2.cvtColor(np.array(texture), cv2.COLOR_RGB2BGR)
                        texture_cv = cv2.resize(texture_cv, (width, height))
                        textures.append(texture_cv)
                    
                    # Blend the textures
                    texture_cv = cv2.addWeighted(textures[0], 0.6, textures[1], 0.4, 0)
                    
                    # Enhance metallic appearance
                    # Increase contrast
                    texture_cv = cv2.convertScaleAbs(texture_cv, alpha=1.3, beta=0)
                    
                    # Apply color grading for metallic look
                    metallic_color = np.array([220, 220, 220])  # Silvery color
                    texture_cv = cv2.addWeighted(texture_cv, 0.7, np.full_like(texture_cv, metallic_color), 0.3, 0)
                    
                    # Add specular highlights
                    highlights = cv2.GaussianBlur(texture_cv, (0, 0), 10)
                    highlights = cv2.convertScaleAbs(highlights, alpha=1.5, beta=50)
                    texture_cv = cv2.addWeighted(texture_cv, 0.7, highlights, 0.3, 0)
                    
                    # Create a gradient overlay for metallic sheen
                    gradient = np.linspace(0, 1, width).reshape(1, -1)
                    gradient = np.tile(gradient, (height, 1))
                    gradient = (gradient * 255).astype(np.uint8)
                    gradient = cv2.GaussianBlur(gradient, (99, 99), 0)
                    gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
                    
                    # Apply gradient overlay
                    texture_cv = cv2.addWeighted(texture_cv, 0.8, gradient, 0.2, 0)
                    
                    # Apply the texture using multiple blending modes
                    mask_blur = cv2.GaussianBlur(table_top_mask, (9, 9), 0)
                    
                    # First pass: Normal clone
                    result1 = cv2.seamlessClone(
                        texture_cv,
                        original_img,
                        mask_blur,
                        (width//2, height//4),
                        cv2.NORMAL_CLONE
                    )
                    
                    # Second pass: Mixed clone for better detail preservation
                    original_img = cv2.seamlessClone(
                        texture_cv,
                        result1,
                        mask_blur,
                        (width//2, height//4),
                        cv2.MIXED_CLONE
                    )

            except Exception as e:
                print(f"Error processing table top: {e}")

        # Process table legs
        if np.any(table_legs_mask):
            prompt = "ultra detailed photograph of sleek chrome table leg, pure metal surface, polished steel cylinder, modern metallic furniture part, glossy chrome finish, industrial design, professional studio lighting, 8k uhd"
            try:
                with torch.autocast(device):
                    # Generate multiple textures and blend them
                    textures = []
                    for _ in range(2):  # Generate 2 variations
                        texture = pipe(
                            prompt=prompt,
                            negative_prompt="wood, wooden, grainy, organic, natural, warm colors, brown, beige, painted, plastic",
                            height=768,
                            width=768,
                            guidance_scale=15.0,
                            num_inference_steps=50
                        ).images[0]
                        texture_cv = cv2.cvtColor(np.array(texture), cv2.COLOR_RGB2BGR)
                        texture_cv = cv2.resize(texture_cv, (width, height))
                        textures.append(texture_cv)
                    
                    # Blend the textures
                    texture_cv = cv2.addWeighted(textures[0], 0.6, textures[1], 0.4, 0)
                    
                    # Enhance metallic appearance
                    texture_cv = cv2.convertScaleAbs(texture_cv, alpha=1.3, beta=0)
                    
                    # Apply chrome-like color grading
                    chrome_color = np.array([240, 240, 240])  # Bright chrome color
                    texture_cv = cv2.addWeighted(texture_cv, 0.7, np.full_like(texture_cv, chrome_color), 0.3, 0)
                    
                    # Add specular highlights
                    highlights = cv2.GaussianBlur(texture_cv, (0, 0), 10)
                    highlights = cv2.convertScaleAbs(highlights, alpha=1.5, beta=50)
                    texture_cv = cv2.addWeighted(texture_cv, 0.7, highlights, 0.3, 0)
                    
                    # Create vertical gradient for leg sheen
                    gradient = np.linspace(0, 1, height).reshape(-1, 1)
                    gradient = np.tile(gradient, (1, width))
                    gradient = (gradient * 255).astype(np.uint8)
                    gradient = cv2.GaussianBlur(gradient, (99, 99), 0)
                    gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
                    
                    # Apply gradient overlay
                    texture_cv = cv2.addWeighted(texture_cv, 0.8, gradient, 0.2, 0)
                    
                    # Apply the texture using multiple blending modes
                    mask_blur = cv2.GaussianBlur(table_legs_mask, (9, 9), 0)
                    
                    # First pass: Normal clone
                    result1 = cv2.seamlessClone(
                        texture_cv,
                        original_img,
                        mask_blur,
                        (width//2, height*3//4),
                        cv2.NORMAL_CLONE
                    )
                    
                    # Second pass: Mixed clone for better detail preservation
                    original_img = cv2.seamlessClone(
                        texture_cv,
                        result1,
                        mask_blur,
                        (width//2, height*3//4),
                        cv2.MIXED_CLONE
                    )

            except Exception as e:
                print(f"Error processing table legs: {e}")

        # Final image enhancement
        # Increase overall contrast and brightness
        original_img = cv2.convertScaleAbs(original_img, alpha=1.2, beta=10)
        # Add final metallic sheen
        metallic_overlay = cv2.GaussianBlur(original_img, (0, 0), 20)
        original_img = cv2.addWeighted(original_img, 0.8, metallic_overlay, 0.2, 0)

        print("Saving result...")
        cv2.imwrite("textured_object.jpg", original_img)
        
        print("Displaying result (press any key within 30 seconds to close)")
        cv2.imshow("Textured Result", original_img)
        key = cv2.waitKey(30000)  # Wait for 30 seconds max
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        print("Cleanup completed")

except Exception as e:
    print(f"An error occurred: {e}")
    cv2.destroyAllWindows()
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()