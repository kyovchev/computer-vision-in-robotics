"""
Bolt detection on Jetson Xavier NX using ONNX Runtime with CUDA execution provider.

TensorRT was tried and dropped: Faster R-CNN's RPN contains proposal-selection
ops (top-k gather, NMS-adjacent reduce) that TensorRT's graph partitioner can't
statically shape-infer, regardless of fixed vs dynamic input shapes. CUDA EP
runs the whole graph on GPU without that limitation.

Setup on the Xavier NX (JetPack 5.x):
    pip3 install onnxruntime-gpu  # matching wheel for your JetPack version

Usage:
    python3 detect_cuda.py --model bolt_model.onnx --csi
    python3 detect_cuda.py --model bolt_model.onnx --usb 0
    python3 detect_cuda.py --model bolt_model.onnx --image test.jpg
"""

import argparse
import time
import numpy as np
import cv2
import onnxruntime as ort


CLASS_NAMES = {1: "bolt", 2: "washer", 3: "nut"}
CLASS_COLORS = {1: (0, 255, 0), 2: (255, 0, 255), 3: (255, 255, 0)}


def gstreamer_pipeline(width=1280, height=720, fps=30, flip_method=0):
    """CSI camera pipeline for the Jetson's onboard camera connector."""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


class BoltDetector:
    def __init__(self, onnx_path, confidence_threshold=0.5, fixed_size=None):
        self.confidence_threshold = confidence_threshold
        self.fixed_size = fixed_size  # (height, width) or None for dynamic-shape models

        print("Loading model with CUDA execution provider...")
        self.session = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        active = self.session.get_providers()
        print(f"Active providers: {active}")
        if "CUDAExecutionProvider" not in active:
            print("WARNING: CUDA EP not active -- running on CPU only, will be very slow. "
                  "Check your onnxruntime-gpu install.")

        print("Warming up...")
        h, w = self.fixed_size if self.fixed_size else (480, 640)
        dummy = np.zeros((3, h, w), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
        print("Ready.")

    def preprocess(self, frame_bgr):
        if self.fixed_size:
            h, w = self.fixed_size
            frame_bgr = cv2.resize(frame_bgr, (w, h))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(chw)

    def detect(self, frame_bgr):
        """
        Returns (boxes, scores, labels), or None if this frame couldn't be
        processed (bad decode, inference error, etc.) -- callers should skip
        the frame rather than crash the whole run.
        """
        try:
            orig_h, orig_w = frame_bgr.shape[:2]
            input_tensor = self.preprocess(frame_bgr)
            boxes, labels, scores = self.session.run(None, {self.input_name: input_tensor})
        except Exception as e:
            print(f"Skipping frame -- inference failed: {type(e).__name__}: {e}")
            return None

        mask = scores >= self.confidence_threshold
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        if self.fixed_size and len(boxes) > 0:
            in_h, in_w = self.fixed_size
            scale_x, scale_y = orig_w / in_w, orig_h / in_h
            boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y], dtype=boxes.dtype)

        return boxes, scores, labels

    def draw(self, frame, boxes, scores, labels):
        out = frame.copy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            color = CLASS_COLORS.get(int(label), (0, 255, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{CLASS_NAMES.get(int(label), str(label))}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(out, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out


def run_image(detector, image_path, output_path=None, show=True):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load {image_path} -- skipping (bad path or corrupt file)")
        return

    result_boxes = detector.detect(frame)
    if result_boxes is None:
        print("Frame could not be processed -- nothing to show/save")
        return
    boxes, scores, labels = result_boxes

    result = detector.draw(frame, boxes, scores, labels)
    print(f"Detected {len(boxes)} objects")
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")
    if show:
        cv2.imshow("Bolt Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(detector, cap, window_title="Bolt Detection"):
    print("Press 'q' to quit")
    prev_time = time.time()
    fps = 0.0
    consecutive_bad_frames = 0
    max_consecutive_bad_frames = 30  # bail out if the camera itself seems dead

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_bad_frames += 1
            print(f"Skipping unreadable frame ({consecutive_bad_frames} in a row)")
            if consecutive_bad_frames >= max_consecutive_bad_frames:
                print("Too many consecutive unreadable frames -- stopping "
                      "(check the camera connection/pipeline).")
                break
            continue
        consecutive_bad_frames = 0

        result_boxes = detector.detect(frame)
        if result_boxes is None:
            continue  # inference failed on this frame -- skip and move on
        boxes, scores, labels = result_boxes

        result = detector.draw(frame, boxes, scores, labels)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        counts = {name: int((labels == cls).sum()) for cls, name in CLASS_NAMES.items()}
        overlay = f"FPS: {fps:.1f} | " + " | ".join(f"{k}: {v}" for k, v in counts.items())
        cv2.putText(result, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_title, result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .onnx model")
    parser.add_argument("--image", help="Run on a single image")
    parser.add_argument("--csi", action="store_true", help="Use onboard CSI camera")
    parser.add_argument("--usb", type=int, help="Use USB camera at this device index")
    parser.add_argument("--output", help="Output path (image mode only)")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--fixed-size", type=str, default=None,
                         help="HxW the ONNX model was exported with, e.g. 480x640. "
                              "Only needed if you exported with --fixed-size on the PC side.")
    args = parser.parse_args()

    fixed_size = None
    if args.fixed_size:
        h, w = (int(x) for x in args.fixed_size.lower().split("x"))
        fixed_size = (h, w)

    detector = BoltDetector(args.model, confidence_threshold=args.confidence, fixed_size=fixed_size)

    if args.image:
        run_image(detector, args.image, args.output, show=not args.no_show)
    elif args.csi:
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("Could not open CSI camera pipeline")
            return
        run_stream(detector, cap)
    elif args.usb is not None:
        cap = cv2.VideoCapture(args.usb)
        if not cap.isOpened():
            print(f"Could not open USB camera at index {args.usb}")
            return
        run_stream(detector, cap)
    else:
        print("Specify --image, --csi, or --usb <index>")


if __name__ == "__main__":
    main()