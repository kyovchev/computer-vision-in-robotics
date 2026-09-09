"""
Export the trained Faster R-CNN bolt detector to ONNX.

Run this on your PC (or wherever the .pth checkpoint lives) BEFORE moving
anything to the Xavier NX. Faster R-CNN is a two-stage detector with
RoIAlign + NMS baked into the graph, so we need opset >= 11 for those ops
to trace correctly.

Usage:
    python export_to_onnx.py --model bolt_model.pth --output bolt_model.onnx
"""

import argparse
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_model(model_path, num_classes=4, device="cpu",
               rpn_post_nms_top_n_test=300, box_detections_per_img=30, box_score_thresh=0.3):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        # default 1000 -- proposals kept after RPN NMS
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        # default 100 -- max final detections per image
        box_detections_per_img=box_detections_per_img,
        # default 0.05 -- discard low-score boxes earlier
        box_score_thresh=box_score_thresh,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def export(model_path, output_path, num_classes=4, img_size=(3, 480, 640), opset=11, dynamic=True,
           rpn_post_nms_top_n_test=300, box_detections_per_img=30, box_score_thresh=0.3):
    device = "cpu"  # export on CPU; TensorRT/ORT will handle GPU at inference time
    model = load_model(model_path, num_classes=num_classes, device=device,
                       rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                       box_detections_per_img=box_detections_per_img,
                       box_score_thresh=box_score_thresh)

    dummy_input = [torch.randn(*img_size, device=device)]

    print(f"Exporting to ONNX (opset {opset}, dynamic={dynamic})...")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input"],
        output_names=["boxes", "labels", "scores"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "input": {1: "height", 2: "width"},  # allow variable image size
        } if dynamic else None,
        dynamo=False,  # force the legacy TorchScript-based exporter -- the
                       # newer dynamo exporter (default in recent PyTorch)
                       # can't trace Faster R-CNN's dynamic image-padding
                       # logic in transform.py's batch_images()
    )
    print(f"Saved: {output_path}")
    if not dynamic:
        print(
            f"\nExported with FIXED input size {img_size[1]}x{img_size[2]}. "
            "Every image fed to this model at inference time must be resized "
            "to exactly this size first (cv2.resize before preprocessing)."
        )
    print(
        "\nNote: torchvision's detection models export boxes/labels/scores as "
        "variable-length outputs since the number of detections isn't fixed. "
        "This is normal for Faster R-CNN and is handled fine by ONNX Runtime."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--output", default="bolt_model.onnx",
                        help="Output ONNX path")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--fixed-size", type=str, default=None,
                        help="Export with a fixed HxW instead of dynamic shapes, "
                             "e.g. --fixed-size 480x640. Use this if dynamic-shape "
                             "export gives TensorRT trouble downstream.")
    parser.add_argument("--max-proposals", type=int, default=300,
                        help="rpn_post_nms_top_n_test -- proposals kept after RPN NMS. "
                             "Lower = faster box head, less recall on crowded scenes. "
                             "Default 300 (torchvision default is 1000).")
    parser.add_argument("--max-detections", type=int, default=30,
                        help="box_detections_per_img -- max final detections per image. "
                             "Set close to the real max object count in your scenes. "
                             "Default 30 (torchvision default is 100).")
    parser.add_argument("--score-thresh", type=float, default=0.3,
                        help="box_score_thresh -- discard low-confidence boxes earlier "
                             "in the pipeline, before the expensive NMS/gather steps. "
                             "Default 0.3 (torchvision default is 0.05).")
    args = parser.parse_args()

    kwargs = dict(
        num_classes=args.num_classes, opset=args.opset,
        rpn_post_nms_top_n_test=args.max_proposals,
        box_detections_per_img=args.max_detections,
        box_score_thresh=args.score_thresh,
    )
    if args.fixed_size:
        h, w = (int(x) for x in args.fixed_size.lower().split("x"))
        export(args.model, args.output, img_size=(
            3, h, w), dynamic=False, **kwargs)
    else:
        export(args.model, args.output, dynamic=True, **kwargs)
