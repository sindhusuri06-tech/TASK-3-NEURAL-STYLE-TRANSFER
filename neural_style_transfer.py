import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load images
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = tf.image.resize(img, (256, 256))
    img = img[tf.newaxis, :]
    return img

content_image = load_image("content.jpg")
style_image = load_image("style.jpg")

# Load pre-trained style transfer model
model = hub.load(
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
)

# Apply style transfer
stylized_image = model(content_image, style_image)[0]

# Save output image
output = stylized_image.numpy()[0] * 255
output = output.astype(np.uint8)
output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
cv2.imwrite("output.jpg", output)

print("Neural Style Transfer completed successfully!")
