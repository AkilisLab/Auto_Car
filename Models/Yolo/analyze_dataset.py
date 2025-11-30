import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

def analyze_dataset_quality():
    """Analyze the current dataset for quality issues"""
    
    # Load data configuration
    with open('stickman_pose/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Analyze class distribution
    def count_classes(labels_dir):
        class_counts = [0, 0, 0, 0]  # fall, run, stand, walk
        total_keypoints = []
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class + bbox
                        class_id = int(parts[0])
                        if 0 <= class_id < 4:
                            class_counts[class_id] += 1
                        
                        # Count keypoints (after bbox: x,y,v for each keypoint)
                        keypoint_data = parts[5:]  # Skip class and bbox
                        if len(keypoint_data) >= 39:  # 13 keypoints * 3 values
                            keypoints = []
                            for i in range(0, 39, 3):
                                x, y, v = float(keypoint_data[i]), float(keypoint_data[i+1]), int(keypoint_data[i+2])
                                keypoints.append((x, y, v))
                            total_keypoints.append(keypoints)
        
        return class_counts, total_keypoints
    
    # Analyze training set
    train_counts, train_keypoints = count_classes('stickman_pose/train/labels')
    val_counts, val_keypoints = count_classes('stickman_pose/valid/labels')
    
    print("Dataset Analysis Report")
    print("=" * 50)
    print(f"Training set class distribution: {train_counts}")
    print(f"Validation set class distribution: {val_counts}")
    print(f"Class names: {data_config['names']}")
    
    # Check for class imbalance
    total_train = sum(train_counts)
    total_val = sum(val_counts)
    
    print(f"\nTotal training samples: {total_train}")
    print(f"Total validation samples: {total_val}")
    
    if total_train < 200:
        print("⚠️  WARNING: Very small training dataset. Recommend at least 500 samples per class.")
    
    # Check class balance
    if total_train > 0:
        max_class = max(train_counts)
        min_class = min([c for c in train_counts if c > 0])
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        if imbalance_ratio > 3:
            print(f"⚠️  WARNING: High class imbalance (ratio: {imbalance_ratio:.2f}). Consider balancing classes.")
    
    # Analyze keypoint quality
    def analyze_keypoints(keypoints, set_name):
        if not keypoints:
            print(f"No keypoints found in {set_name} set")
            return
            
        visible_counts = []
        for kpts in keypoints:
            visible = sum(1 for _, _, v in kpts if v == 2)  # v=2 means visible
            visible_counts.append(visible)
        
        avg_visible = np.mean(visible_counts)
        print(f"\n{set_name} keypoint analysis:")
        print(f"Average visible keypoints per person: {avg_visible:.1f}/13")
        print(f"Minimum visible keypoints: {min(visible_counts) if visible_counts else 0}")
        print(f"Maximum visible keypoints: {max(visible_counts) if visible_counts else 0}")
        
        if avg_visible < 8:
            print("⚠️  WARNING: Low average visible keypoints. Consider improving annotation quality.")
    
    analyze_keypoints(train_keypoints, "Training")
    analyze_keypoints(val_keypoints, "Validation")
    
    # Recommendations
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("=" * 50)
    
    if total_train < 500:
        print("1. 🔴 CRITICAL: Increase dataset size to at least 500 samples per class")
    
    if max(train_counts) / min([c for c in train_counts if c > 0]) > 2:
        print("2. 🟡 MEDIUM: Balance class distribution")
    
    if len(train_keypoints) > 0 and np.mean([sum(1 for _, _, v in kpts if v == 2) for kpts in train_keypoints]) < 10:
        print("3. 🟡 MEDIUM: Improve keypoint annotation quality")
    
    print("4. 🟢 RECOMMENDED: Add background/negative samples (15-25% of dataset)")
    print("5. 🟢 RECOMMENDED: Use data augmentation carefully to preserve pose relationships")

def create_improved_data_yaml():
    """Create an improved data.yaml with better configuration"""
    
    improved_config = {
        'train': '../train/images',
        'val': '../valid/images', 
        'test': '../test/images',
        
        # Pose estimation specific
        'kpt_shape': [13, 3],  # 13 keypoints, 3 values each (x,y,visibility)
        'flip_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # No flipping for keypoints
        
        # Classes
        'nc': 4,
        'names': ['person fall', 'person run', 'person stand', 'person walk'],
        
        # Additional metadata for better training
        'download': None,  # Set if data needs to be downloaded
        'yaml_file': 'data_improved.yaml'
    }
    
    with open('stickman_pose/data_improved.yaml', 'w') as f:
        yaml.dump(improved_config, f, default_flow_style=False)
    
    print("Created improved data configuration: stickman_pose/data_improved.yaml")

def visualize_annotations(num_samples=5):
    """Visualize a few samples with annotations to check quality"""
    
    import random
    
    images_dir = 'stickman_pose/train/images'
    labels_dir = 'stickman_pose/train/labels'
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("Image or label directory not found")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    fig, axes = plt.subplots(1, len(sample_files), figsize=(5*len(sample_files), 5))
    if len(sample_files) == 1:
        axes = [axes]
    
    # CORRECT keypoint connections based on actual schema:
    # kp0: head, kp1: neck, kp2,3,4: left shoulder/elbow/hand
    # kp5,6,7: right shoulder/elbow/hand, kp8: hip
    # kp9,10: left knee/foot, kp11,12: right knee/foot
    keypoint_connections = [
        # Head and neck
        (0, 1),   # Head to neck
        
        # Torso (neck to hip)
        (1, 8),   # Neck to hip
        
        # Left arm chain
        (1, 2),   # Neck to left shoulder
        (2, 3),   # Left shoulder to left elbow
        (3, 4),   # Left elbow to left hand
        
        # Right arm chain
        (1, 5),   # Neck to right shoulder
        (5, 6),   # Right shoulder to right elbow
        (6, 7),   # Right elbow to right hand
        
        # Left leg chain
        (8, 9),   # Hip to left knee
        (9, 10),  # Left knee to left foot
        
        # Right leg chain
        (8, 11),  # Hip to right knee
        (11, 12), # Right knee to right foot
    ]
    
    for i, img_file in enumerate(sample_files):
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, img_file.rsplit('.', 1)[0] + '.txt')
        
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Read annotations
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 44:  # class + 4 bbox + 39 keypoints
                    continue
                
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert normalized to pixel coordinates
                x_center *= w
                y_center *= h
                width *= w  
                height *= h
                
                # Draw bounding box
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw keypoints
                keypoint_data = parts[5:44]  # 13 keypoints * 3 values
                keypoints = []
                
                for j in range(0, 39, 3):
                    if j+2 < len(keypoint_data):
                        kx, ky, kv = float(keypoint_data[j]), float(keypoint_data[j+1]), int(keypoint_data[j+2])
                        kx *= w
                        ky *= h
                        keypoints.append((kx, ky, kv))
                        
                        # Draw keypoint if visible
                        if kv == 2:  # visible
                            cv2.circle(img, (int(kx), int(ky)), 3, (0, 255, 0), -1)
                        elif kv == 1:  # occluded
                            cv2.circle(img, (int(kx), int(ky)), 3, (255, 255, 0), -1)
                
                # Draw connections
                for conn in keypoint_connections:
                    if conn[0] < len(keypoints) and conn[1] < len(keypoints):
                        kp1, kp2 = keypoints[conn[0]], keypoints[conn[1]]
                        if kp1[2] > 0 and kp2[2] > 0:  # both keypoints visible
                            cv2.line(img, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), (0, 255, 255), 2)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('annotation_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved annotation visualization to 'annotation_samples.png'")

if __name__ == "__main__":
    print("Analyzing dataset quality...")
    analyze_dataset_quality()
    
    print("\nCreating improved configuration...")
    create_improved_data_yaml()
    
    print("\nGenerating annotation visualizations...")
    try:
        visualize_annotations(3)
    except Exception as e:
        print(f"Could not generate visualizations: {e}")