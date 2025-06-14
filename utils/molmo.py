
from transformers import GenerationConfig
from PIL import Image, ImageDraw
import re


# Extract coordinates from Molmo response
def extract_points(text):
    pattern = r'x\d+="([\d.]+)"\s+y\d+="([\d.]+)"'
    matches = re.findall(pattern, text)
    points = [(float(x), float(y)) for x, y in matches]
    return points


# Get the point of all object of the scene using Molmo
def points_object_detection_molmo(processor, model, image, object_detection_prompt):

    inputs = processor.process(
        images=[image],
        text=object_detection_prompt,
    )
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    inputs["images"] = inputs["images"].to(model.dtype)

    # Query to the model
    result = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer,
    )
    generated_tokens = result[0, inputs["input_ids"].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("📝 Molmo output:\n", generated_text)

    points = extract_points(generated_text)
    print(f"📍 Points détectés : {points}")

    
    width, height = image.size

    points_scaled = [(
            x / 100 * width,
            y / 100 * height
    ) for (x, y) in points]

    return points_scaled


# Visualize the result of Molmo 
def draw_points_on_image(image, points, color="red", radius=6):
    draw = ImageDraw.Draw(image)

    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image
