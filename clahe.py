"""
VGG-19 based Grayscale Image Colorization
Post-processing with CLAHE (Applied AFTER colorization)

Author: Dr. Gouranga Mandal
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------
# PARAMETERS
# ---------------------------------------------------
IMG_SIZE = 224
INPUT_IMAGE = "sampleimage.jpg"
WEIGHTS_PATH = "colorization_vgg19_weights.h5"
OUTPUT_IMAGE = "final_colorized_clahe.jpg"

# ---------------------------------------------------
# STEP 1: LOAD GRAYSCALE IMAGE
# ---------------------------------------------------
gray = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError("Input image not found!")

gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

L = gray.astype("float32") / 255.0
L = L * 100.0                         # LAB L-channel range
L_input = L.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# ---------------------------------------------------
# STEP 2: BUILD VGG-19 COLORIZATION MODEL
# ---------------------------------------------------
vgg = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
vgg.trainable = False

vgg_features = Model(
    inputs=vgg.input,
    outputs=vgg.get_layer("block4_conv4").output
)

input_L = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Convert L → pseudo RGB
x = Conv2D(3, (1, 1), padding="same")(input_L)

features = vgg_features(x)

x = Conv2D(256, (3,3), activation="relu", padding="same")(features)
x = UpSampling2D((2,2))(x)

x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = UpSampling2D((2,2))(x)

x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = UpSampling2D((2,2))(x)

x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
x = UpSampling2D((2,2))(x)

output_ab = Conv2D(
    2, (3,3),
    activation="tanh",
    padding="same"
)(x)

model = Model(input_L, output_ab)
model.compile(optimizer=Adam(1e-4), loss="mse")

# ---------------------------------------------------
# STEP 3: LOAD TRAINED WEIGHTS
# ---------------------------------------------------
model.load_weights(WEIGHTS_PATH)

# ---------------------------------------------------
# STEP 4: COLORIZE IMAGE USING VGG-19
# ---------------------------------------------------
pred_ab = model.predict(L_input)[0]
pred_ab = pred_ab * 128.0

lab_output = np.zeros((IMG_SIZE, IMG_SIZE, 3))
lab_output[:,:,0] = L
lab_output[:,:,1:] = pred_ab

rgb_colorized = lab2rgb(lab_output)
rgb_colorized_uint8 = (rgb_colorized * 255).astype("uint8")

# ---------------------------------------------------
# STEP 5: POST-PROCESSING WITH CLAHE (CORRECT PLACE)
# ---------------------------------------------------
lab_post = rgb2lab(rgb_colorized_uint8)
L_post = lab_post[:,:,0].astype("uint8")

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

L_enhanced = clahe.apply(L_post)
lab_post[:,:,0] = L_enhanced

final_rgb = lab2rgb(lab_post)
final_rgb = (final_rgb * 255).astype("uint8")

# ---------------------------------------------------
# STEP 6: SAVE OUTPUT
# ---------------------------------------------------
cv2.imwrite(
    OUTPUT_IMAGE,
    cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
)

# ---------------------------------------------------
# STEP 7: DISPLAY RESULTS
# ---------------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Input Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("VGG-19 Colorized")
plt.imshow(rgb_colorized_uint8)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Colorized + CLAHE (Post)")
plt.imshow(final_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()

print("Colorization completed successfully.")
print(f"Output saved as: {OUTPUT_IMAGE}")
