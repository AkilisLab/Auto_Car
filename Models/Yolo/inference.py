#!/usr/bin/env python3
"""
Stickman Pose Inference Script
==============================

Runs inference on test images using trained pose estimation model.
Features:
- Automatic model detection from latest training runs
- Comprehensive pose visualization with anatomical skeleton
- Confidence scoring and pose quality metrics
- Batch processing with progress tracking
"""

from ultralytics import YOLO
import os
import cv2
import numpy as np
import glob
from pathlib import Path
import time

# Configuration
class Config:
    # Auto-detect best model from training runs
    POSSIBLE_MODEL_PATHS = [
        './runs_stickman_pose_optimized/small_dataset_training/weights/best.pt'
    ]
    
    # Test data paths
    TEST_IMAGES_DIR = './stickman_pose/test/images'
    VALID_IMAGES_DIR = './stickman_pose/valid/images'  # Alternative test set
    
    # Output configuration
    OUTPUT_DIR = './inference_results'
    CONFIDENCE_THRESHOLD = 0.25
    
    # Pose visualization settings
    KEYPOINT_COLORS = {
        'visible': (0, 255, 0),    # Green for visible keypoints
        'occluded': (0, 165, 255), # Orange for occluded
        'skeleton': (255, 0, 0),   # Red for skeleton
        'bbox': (0, 255, 255)      # Yellow for bounding box
    }
    
    # Anatomically correct keypoint connections
    POSE_CONNECTIONS = [
        (0, 1),   # Head to neck
        (1, 8),   # Neck to hip (spine)
        (1, 2),   # Neck to left shoulder
        (2, 3),   # Left shoulder to elbow
        (3, 4),   # Left elbow to hand
        (1, 5),   # Neck to right shoulder
        (5, 6),   # Right shoulder to elbow
        (6, 7),   # Right elbow to hand
        (8, 9),   # Hip to left knee
        (9, 10),  # Left knee to foot
        (8, 11),  # Hip to right knee
        (11, 12), # Right knee to foot
    ]
    
    CLASS_NAMES = ['FALL', 'RUN', 'STAND', 'WALK']

def find_best_model():
    """Auto-detect the best trained model"""
    print("🔍 Searching for trained models...")
    
    for model_path in Config.POSSIBLE_MODEL_PATHS:
        if os.path.exists(model_path):
            print(f"✅ Found model: {model_path}")
            return model_path
    
    # Fallback: search all runs directories
    run_dirs = glob.glob('./runs_stickman_pose*/*/weights/best.pt')
    if run_dirs:
        latest_model = max(run_dirs, key=os.path.getctime)
        print(f"✅ Found latest model: {latest_model}")
        return latest_model
    
    print("❌ No trained models found! Please run training first.")
    return None

def get_test_images():
    """Get list of test images from available directories"""
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    
    # Check test directory first
    if os.path.exists(Config.TEST_IMAGES_DIR):
        print(f"📁 Using test images from: {Config.TEST_IMAGES_DIR}")
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(Config.TEST_IMAGES_DIR, ext)))
    
    # Fallback to validation directory
    elif os.path.exists(Config.VALID_IMAGES_DIR):
        print(f"📁 Using validation images from: {Config.VALID_IMAGES_DIR}")
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(Config.VALID_IMAGES_DIR, ext)))
    
    if not image_files:
        print("❌ No test images found!")
        return []
    
    print(f"📊 Found {len(image_files)} test images")
    return image_files

def draw_pose_with_skeleton(image, results):
    """Draw pose with anatomically correct skeleton"""
    img = image.copy()
    
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        
        if boxes is not None and keypoints is not None:
            for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                if conf < Config.CONFIDENCE_THRESHOLD:
                    continue
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), Config.KEYPOINT_COLORS['bbox'], 2)
                
                # Draw class label
                label = f"{Config.CLASS_NAMES[cls]} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), Config.KEYPOINT_COLORS['bbox'], -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # Get keypoints
                kpts_xy = kpts.xy[0].cpu().numpy()  # Shape: (num_keypoints, 2)
                kpts_conf = kpts.conf[0].cpu().numpy()  # Shape: (num_keypoints,)
                
                # Draw skeleton connections first
                for connection in Config.POSE_CONNECTIONS:
                    if (connection[0] < len(kpts_xy) and connection[1] < len(kpts_xy) and
                        kpts_conf[connection[0]] > 0.1 and kpts_conf[connection[1]] > 0.1):
                        
                        pt1 = tuple(kpts_xy[connection[0]].astype(int))
                        pt2 = tuple(kpts_xy[connection[1]].astype(int))
                        cv2.line(img, pt1, pt2, Config.KEYPOINT_COLORS['skeleton'], 3)
                
                # Draw keypoints on top
                for j, (kpt, conf_kpt) in enumerate(zip(kpts_xy, kpts_conf)):
                    if conf_kpt > 0.1:  # Only draw confident keypoints
                        x, y = int(kpt[0]), int(kpt[1])
                        
                        # Choose color based on confidence
                        color = Config.KEYPOINT_COLORS['visible'] if conf_kpt > 0.5 else Config.KEYPOINT_COLORS['occluded']
                        
                        # Draw keypoint
                        cv2.circle(img, (x, y), 6, (0, 0, 0), -1)  # Black border
                        cv2.circle(img, (x, y), 4, color, -1)      # Colored center
    
    return img

def run_inference():
    """Run comprehensive inference on test images"""
    print("🚀 Starting Pose Inference...")
    print("=" * 50)
    
    # Find model
    model_path = find_best_model()
    if not model_path:
        return
    
    # Load model
    print(f"📥 Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get test images
    image_files = get_test_images()
    if not image_files:
        return
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_path = Path(Config.OUTPUT_DIR)
    
    # Process images
    print(f"\n🎯 Processing {len(image_files)} images...")
    start_time = time.time()
    
    results_summary = []
    
    for i, img_path in enumerate(image_files, 1):
        try:
            print(f"📸 [{i:2d}/{len(image_files)}] Processing: {Path(img_path).name}")
            
            # Run inference
            results = model(img_path, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
            
            # Load original image for custom visualization
            original_img = cv2.imread(img_path)
            if original_img is None:
                print(f"   ⚠️ Failed to load image: {img_path}")
                continue
            
            # Create custom visualization
            viz_img = draw_pose_with_skeleton(original_img, results)
            
            # Save result
            output_filename = f"pose_{Path(img_path).stem}.jpg"
            output_file = output_path / output_filename
            cv2.imwrite(str(output_file), viz_img)
            
            # Collect statistics
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            avg_conf = float(results[0].boxes.conf.mean()) if num_detections > 0 else 0.0
            
            results_summary.append({
                'image': Path(img_path).name,
                'detections': num_detections,
                'avg_confidence': avg_conf
            })
            
            print(f"   ✅ Saved: {output_filename} ({num_detections} poses, avg conf: {avg_conf:.3f})")
            
        except Exception as e:
            print(f"   ❌ Error processing {img_path}: {e}")
            continue
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print("📊 INFERENCE COMPLETE!")
    print("=" * 50)
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📁 Results saved to: {Config.OUTPUT_DIR}")
    print(f"🎯 Average processing time: {total_time/len(image_files):.3f} sec/image")
    
    # Statistics
    if results_summary:
        total_detections = sum(r['detections'] for r in results_summary)
        avg_detections = total_detections / len(results_summary)
        overall_conf = np.mean([r['avg_confidence'] for r in results_summary if r['avg_confidence'] > 0])
        
        print(f"\n📈 Detection Statistics:")
        print(f"   Total poses detected: {total_detections}")
        print(f"   Average poses per image: {avg_detections:.1f}")
        print(f"   Overall confidence: {overall_conf:.3f}")
    
    print(f"\n🎨 View results with anatomically correct pose skeletons!")
    print("=" * 50)

if __name__ == "__main__":
    run_inference()
