1. Export model with `python export_to_onnx.py --model models/best_model.pth --output models/bolt_model.onnx --fixed-size 320x416 --max-proposals 300 --max-detections 30 --score-thresh 0.3`

2. Validate model with python `test_onnx.py`

3. Copy the exported model to Jetson Xavier NX

4. Execute `detect_cuda.py` with `run_image.sh` or `run_usb.sh` on the Jetson Xavier NX
