import cv2
import time
import os

# Params
DEVICE_ID = 1
PATH = './capture'
PARAMS_FILE = './capture/camera_params.txt'

# --- Trackbar ranges — tweak to match what your driver actually reports ---
# log2 scale on many UVC/DirectShow drivers
EXPOSURE_MIN, EXPOSURE_MAX = -13, 0
BRIGHTNESS_MIN, BRIGHTNESS_MAX = 0, 255
CONTRAST_MIN, CONTRAST_MAX = 0, 255
SATURATION_MIN, SATURATION_MAX = 0, 255
SHARPNESS_MIN, SHARPNESS_MAX = 0, 255


def nothing(x):
    pass


def make_trackbar(name, window, min_val, max_val, start_val):
    """Trackbars only accept non-negative ints, so shift ranges that go negative."""
    cv2.createTrackbar(name, window, start_val - min_val,
                       max_val - min_val, nothing)


def read_trackbar(name, window, min_val):
    return cv2.getTrackbarPos(name, window) + min_val


os.makedirs(PATH, exist_ok=True)

cap = cv2.VideoCapture(DEVICE_ID, cv2.CAP_DSHOW)  # use cv2.CAP_V4L2 on Linux

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print('Cannot open camera!')
    exit()

# Start in manual exposure/focus mode so sliders take effect immediately
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Controls', 400, 800)

# --- Exposure / focus ---
cv2.createTrackbar('Auto Exposure', 'Controls', 1,
                   1, nothing)   # 0=manual, 1=auto
make_trackbar('Exposure', 'Controls', EXPOSURE_MIN, EXPOSURE_MAX, -6)

# --- Image settings ---
make_trackbar('Brightness', 'Controls', BRIGHTNESS_MIN, BRIGHTNESS_MAX, 50)
make_trackbar('Contrast', 'Controls', CONTRAST_MIN, CONTRAST_MAX, 50)
make_trackbar('Saturation', 'Controls', SATURATION_MIN, SATURATION_MAX, 50)
make_trackbar('Sharpness', 'Controls', SHARPNESS_MIN, SHARPNESS_MAX, 50)

print("Controls: C=Capture, S=Save params, Esc=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print('Cannot retrieve camera frame!')
        break

    # --- Exposure ---
    auto_exp = cv2.getTrackbarPos('Auto Exposure', 'Controls')
    exposure = read_trackbar('Exposure', 'Controls', EXPOSURE_MIN)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exp else 0.25)
    if not auto_exp:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    # --- Image settings ---
    brightness = read_trackbar('Brightness', 'Controls', BRIGHTNESS_MIN)
    contrast = read_trackbar('Contrast', 'Controls', CONTRAST_MIN)
    saturation = read_trackbar('Saturation', 'Controls', SATURATION_MIN)
    sharpness = read_trackbar('Sharpness', 'Controls', SHARPNESS_MIN)

    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)

    frame2 = frame.copy()
    cv2.putText(frame2, 'C=Capture, S=Save params, Esc=Quit', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Camera', cv2.resize(frame2, (1200, 900)))

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = f'{PATH}/capture_{int(time.time())}.png'
        cv2.imwrite(filename, frame)
        print(f'Image saved as {filename}')

    elif key == ord('s'):
        with open(PARAMS_FILE, 'a') as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"AutoExposure={auto_exp} Exposure={exposure} "
                f"Brightness={brightness} Contrast={contrast} "
                f"Saturation={saturation} Sharpness={sharpness}\n"
            )
        print(f'Parameters saved to {PARAMS_FILE}')

    elif key == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
