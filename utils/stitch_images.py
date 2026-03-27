from PIL import Image, ImageDraw, ImageFont
import os

images = []
labels = [
    "G0: Baseline (0-shot)", "G1: Waterfall (Linear)", 
    "G2: AgentCoder (Loop)", "G3: MapCoder (Cycle)", 
    "G4: Parallel Judge", "G5: Adversarial Debate", 
    "G6: Hierarchical Setup"
]

for i in range(1, 8):
    file_name = f"graph_structure-{i}.png"
    if os.path.exists(file_name):
        images.append(Image.open(file_name))

if images:
    # Set padding and text height space
    padding = 20
    text_height = 40
    
    total_height = sum(img.height + text_height + padding for img in images) + padding
    max_width = max(img.width for img in images) + padding * 2
    
    # Create white canvas
    composite = Image.new('RGB', (max_width, total_height), 'white')
    draw = ImageDraw.Draw(composite)
    
    y_offset = padding
    for i, img in enumerate(images):
        # Draw label
        draw.text((padding, y_offset), labels[i], fill="black")
        y_offset += text_height
        
        # Paste image centered
        x_offset = padding + (max_width - padding*2 - img.width) // 2
        composite.paste(img, (x_offset, y_offset))
        y_offset += img.height + padding
        
    composite.save("graph_structure.png")
    print("Combined image saved to graph_structure.png")
else:
    print("No images found to combine.")
