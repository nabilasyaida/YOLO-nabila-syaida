from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1. Load model
model = YOLO("best.pt")

# 2. Predict image
results = model("WHITEHEADS DAN PUSTULE.jpg")[0]

# 3. Count classes
class_ids = results.boxes.cls.cpu().numpy()
class_names = results.names
counts = Counter(class_ids)

# 4. Get the image with bounding boxes
img_with_boxes = results.plot()

# 5. Convert to PIL Image
img_pil = Image.fromarray(img_with_boxes)

# 6. Load background image
background = Image.open("bccbd03b-a912-417e-ae18-7918fda5d67e.jpg").convert("RGB")

# Resize background to match detection image (optional)
background = background.resize(img_pil.size)

# 7. Paste detection result onto background with alpha mask if needed
background.paste(img_pil, (0, 0), img_pil if img_pil.mode == 'RGBA' else None)

# 8. Draw class counts on the combined image
draw = ImageDraw.Draw(background)
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

y_offset = 30
for i, (class_id, count) in enumerate(counts.items()):
    name = class_names[int(class_id)]
    label = f"{name}: {count}"
    draw.text((10, y_offset + i * 30), label, font=font, fill=(255, 0, 0))

# 9. Save and show
background.save("output_with_background.jpg")

plt.imshow(background)
plt.axis('off')
plt.show()
