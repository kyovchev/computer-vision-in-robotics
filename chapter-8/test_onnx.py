import cv2
import numpy as np
import onnxruntime as ort


ONNX_PATH = "models/bolt_model.onnx"
IMAGE_PATH = "image.png"

# Must match the --fixed-size HxW used in export_to_onnx.py
# (e.g. --fixed-size 480x640 -> INPUT_HEIGHT=480, INPUT_WIDTH=640)
INPUT_HEIGHT = 320
INPUT_WIDTH = 416
CONFIDENCE = 0.5


# --------------------------------------------------
# Load ONNX
# --------------------------------------------------

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded")

for inp in session.get_inputs():
    print("Input:", inp.name, inp.shape, inp.type)

for out in session.get_outputs():
    print("Output:", out.name, out.shape, out.type)


# --------------------------------------------------
# Load original image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        f"Could not load image: {IMAGE_PATH}"
    )

original = image.copy()

original_height, original_width = image.shape[:2]

print(
    f"Original image: "
    f"{original_width} x {original_height}"
)


# --------------------------------------------------
# Preprocess
# --------------------------------------------------

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image_resized = cv2.resize(
    image_rgb,
    (INPUT_WIDTH, INPUT_HEIGHT)  # cv2.resize takes (width, height)
)

image_tensor = image_resized.astype(
    np.float32
) / 255.0

# HWC → CHW
# NOTE: no batch dimension here -- the export script passes a single
# CHW tensor (torchvision detection models take a list of per-image
# tensors internally), so the ONNX graph expects shape (3, H, W), not
# (1, 3, H, W). Adding a batch axis will cause a shape mismatch.
image_tensor = np.transpose(
    image_tensor,
    (2, 0, 1)
)


# --------------------------------------------------
# ONNX inference
# --------------------------------------------------

outputs = session.run(
    None,
    {
        "input": image_tensor
    }
)

boxes = outputs[0]
labels = outputs[1]
scores = outputs[2]


print("\nRaw output:")
print("Boxes :", boxes.shape)
print("Labels:", labels.shape)
print("Scores:", scores.shape)


# --------------------------------------------------
# Scale boxes back to original image
# --------------------------------------------------

scale_x = original_width / INPUT_WIDTH
scale_y = original_height / INPUT_HEIGHT


# --------------------------------------------------
# Draw detections
# --------------------------------------------------

for box, label, score in zip(
    boxes,
    labels,
    scores
):

    if score < CONFIDENCE:
        continue

    x1, y1, x2, y2 = box

    # Convert resized-image coordinates
    # back to original image coordinates

    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    # Clamp coordinates
    x1 = max(0, min(x1, original_width - 1))
    y1 = max(0, min(y1, original_height - 1))
    x2 = max(0, min(x2, original_width - 1))
    y2 = max(0, min(y2, original_height - 1))

    # Class name

    label = int(label)

    if label == 1:
        class_name = "bolt"

    elif label == 2:
        class_name = "washer"

    elif label == 3:
        class_name = "nut"

    else:
        class_name = f"class {label}"

    print(
        f"{class_name}: "
        f"{score:.3f} "
        f"[{x1}, {y1}, {x2}, {y2}]"
    )

    # Draw bounding box

    cv2.rectangle(
        original,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Label

    text = f"{class_name}: {score:.2f}"

    cv2.putText(
        original,
        text,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


# --------------------------------------------------
# Display
# --------------------------------------------------

cv2.imshow(
    "ONNX Detection",
    original
)

cv2.waitKey(0)
cv2.destroyAllWindows()