#!/usr/bin/env python3
"""
Stickman Pose Estimation Training Script
========================================

Dataset: 83 training samples, 20 validation samples
Classes: 4 (fall, run, stand, walk)
Keypoints: 13-point custom schema
Challenge: Small dataset requires careful hyperparameter tuning

Optimizations applied:
- Conservative augmentation to preserve pose relationships
- Longer training with early stopping
- Pose-focused loss weights
- Progressive learning strategy
"""

from ultralytics import YOLO
import torch

# Verify GPU availability
print(f"🚀 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")

# Use YOLOv8 pose model for fine-tuning
print("📋 Loading YOLOv8n-pose pretrained model...")
model = YOLO("yolov8n-pose.pt")

print("🎯 Starting optimized training for small pose dataset...")
print("📊 Dataset: 83 train + 20 val samples")
print("🎪 Strategy: Conservative augmentation + Extended training")

results = model.train(
    # --------------------------
    # Dataset Configuration
    # --------------------------
    data="./stickman_pose/data.yaml",
    imgsz=640,
    
    # --------------------------
    # Training Strategy (Optimized for Small Dataset)
    # --------------------------
    epochs=300,        # Extended training for small dataset
    patience=50,       # More patience for small dataset convergence
    batch=4,           # Smaller batch for better gradient updates
    lr0=0.001,         # Conservative initial learning rate
    lrf=0.01,          # Final learning rate factor
    momentum=0.937,    # Standard momentum
    weight_decay=0.0005, # L2 regularization
    
    # --------------------------
    # Hardware & Saving
    # --------------------------
    device=0,
    save=True,
    save_period=10,    # Save less frequently
    project="./runs_stickman_pose_optimized",
    name="small_dataset_training",
    exist_ok=True,
    
    # --------------------------
    # Pose-Specific Loss Weights
    # --------------------------
    pose=12.0,         # Higher pose loss weight
    kobj=1.0,          # Keypoint objectness
    cls=0.5,           # Lower classification weight
    dfl=1.5,           # Distribution focal loss
    
    # --------------------------
    # Background-Robust Augmentation Strategy
    # --------------------------
    degrees=5,         # Minimal rotation (pose-friendly)
    translate=0.05,    # Minimal translation
    scale=0.05,        # Minimal scaling
    shear=0.5,         # Minimal shear
    perspective=0.0,   # No perspective transform
    flipud=0.0,        # No vertical flip
    fliplr=0.5,        # Horizontal flip OK for poses
    
    # Background Diversity Techniques
    mosaic=0.7,        # Higher mosaic - combines different backgrounds
    mixup=0.1,         # Light mixup - helps background generalization
    copy_paste=0.1,    # Copy poses onto different backgrounds
    
    # Enhanced color augmentation for background robustness
    hsv_h=0.02,        # Slight hue variation (backgrounds)
    hsv_s=0.4,         # More saturation change (lighting conditions)
    hsv_v=0.3,         # More brightness change (indoor/outdoor)
    
    # --------------------------
    # Validation & Monitoring
    # --------------------------
    val=True,
    plots=True,        # Generate training plots
    verbose=True,      # Detailed logging
)

# --------------------------
# Post-Training Analysis
# --------------------------
print("\n" + "="*60)
print("🎉 TRAINING COMPLETED!")
print("="*60)

# Training summary
if results:

    if hasattr(results, 'fitness'):
        print(f"🎯 Fitness score: {results.fitness:.4f}")
    if hasattr(results, 'box') and hasattr(results.box, 'map50'):
        print(f"📈 Box mAP50: {results.box.map50:.4f}")
    if hasattr(results, 'box') and hasattr(results.box, 'map'):
        print(f"📈 Box mAP50-95: {results.box.map:.4f}")
    if hasattr(results, 'pose') and hasattr(results.pose, 'map50'):
        print(f"📈 Pose mAP50: {results.pose.map50:.4f}")
    if hasattr(results, 'pose') and hasattr(results.pose, 'map'):
        print(f"📈 Pose mAP50-95: {results.pose.map:.4f}")
    if hasattr(results, 'save_dir'):
        print(f"📁 Results saved to: {results.save_dir}")

# Load best model for validation
print("\n🔍 Loading best trained model for validation...")
best_model = YOLO(results.save_dir / 'weights' / 'best.pt')

# Validate on test set if available
print("🧪 Running validation on best model...")
val_results = best_model.val()

if val_results:
    print(f"📈 Validation Results:")
    print(f"   Box mAP50: {val_results.box.map50:.4f}")
    print(f"   Box mAP50-95: {val_results.box.map:.4f}")
    if hasattr(val_results, 'pose'):
        print(f"   Pose mAP50: {val_results.pose.map50:.4f}")
        print(f"   Pose mAP50-95: {val_results.pose.map:.4f}")

# Export optimized model
print("\n📦 Exporting trained model...")
try:
    export_path = best_model.export(
        format="onnx", 
        imgsz=640, 
        opset=11,
        simplify=True,
        dynamic=False
    )
    print(f"✅ Model exported to: {export_path}")
except Exception as e:
    print(f"⚠️ Export failed: {e}")
    print("Model training completed but export failed.")

print("🎯 Next Steps:")
print("1. Review training plots in the results directory")
print("2. Test the model with inference.py")
print("3. CRITICAL: Add background diversity - see BACKGROUND_DIVERSITY_PLAN.md")
print("4. Collect 500+ images per class with varied backgrounds")
print("5. Add 300+ background-only (no people) images")
print("6. Focus on FALL class - currently 0% detection rate")
print("\\n" + "="*60)
