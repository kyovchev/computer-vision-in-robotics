1. Capture images with [`capture.py`](../chapter-2/capture.py) script from [`chapter-2`](../chapter-2/)

2. Annotate dataset with `python annotate.py --images data/images --output data/annotations.json --split`
Tool controls:
1 - Annotate bottle
2 - Annotate cap
Click and drag - Draw bounding box
R - RESET the current image annotations (reset functionality is still under development)
N - Next image (save the current annotation)
P - Previous image
D - Delete the last bounding box
S - Save and continue
Q - Save and Quit

3. Check for CUDA with `python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"`. If not available install with `python -m pip uninstall torch torchvision torchaudio -y`

4. Train model with `python train.py`

5. Inference with:
```
# Basic
python detect.py --model checkpoints/best_model.pth --image test.jpg

# Save to image
python detect.py --model checkpoints/best_model.pth --image test.jpg --output result.jpg

# Different confidence threshold
python detect.py --model checkpoints/best_model.pth --image test.jpg --confidence 0.7

# Video
python detect.py --model checkpoints/best_model.pth --video input.mp4 --output output.mp4

# Webcam stream
python detect.py --model checkpoints/best_model.pth --webcam
```
