import kagglehub
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
from pathlib import Path

from model import CNN  # Import your CNN class

def explore_dataset_structure(dataset_path, max_depth=3):
    """Explore and print the dataset directory structure"""
    print(f"\nExploring dataset structure at: {dataset_path}")
    dataset_path = Path(dataset_path)
    
    def print_tree(path, prefix="", depth=0, max_depth=max_depth):
        if depth > max_depth:
            return
        
        items = list(path.iterdir())
        items.sort(key=lambda x: (x.is_file(), x.name.lower()))
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and depth < max_depth:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, depth + 1, max_depth)
    
    print_tree(dataset_path)

def get_all_images_from_dataset_flexible(dataset_path):
    """Get all image paths from dataset with flexible structure detection"""
    image_paths = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    dataset_path = Path(dataset_path)
    print(f"\nSearching for images in: {dataset_path}")
    
    # First, let's see what's in the root directory
    root_items = list(dataset_path.iterdir())
    print(f"Items in root directory: {[item.name for item in root_items]}")
    
    # Strategy 1: Look for images directly in subdirectories
    found_structure = False
    for item in root_items:
        if item.is_dir():
            # Check if this directory contains images
            images_in_dir = []
            for file in item.iterdir():
                if file.is_file() and file.suffix.lower() in image_extensions:
                    images_in_dir.append(file)
            
            if images_in_dir:
                print(f"Found {len(images_in_dir)} images in {item.name}")
                for img_file in images_in_dir:
                    image_paths.append((str(img_file), item.name))
                found_structure = True
    
    # Strategy 2: If no class folders found, look for train/test/val structure
    if not found_structure:
        for split_folder in ['train', 'test', 'val', 'training', 'testing', 'validation']:
            split_path = dataset_path / split_folder
            if split_path.exists() and split_path.is_dir():
                print(f"Found {split_folder} folder, exploring...")
                for class_folder in split_path.iterdir():
                    if class_folder.is_dir():
                        images_in_class = []
                        for img_file in class_folder.iterdir():
                            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                                images_in_class.append(img_file)
                        
                        if images_in_class:
                            print(f"Found {len(images_in_class)} images in {split_folder}/{class_folder.name}")
                            for img_file in images_in_class:
                                image_paths.append((str(img_file), class_folder.name))
                            found_structure = True
    
    # Strategy 3: Look for raw-samples folder (common in some Kaggle datasets)
    raw_samples_path = dataset_path / 'raw-img'
    if raw_samples_path.exists():
        print(f"Found raw-img folder, exploring...")
        for class_folder in raw_samples_path.iterdir():
            if class_folder.is_dir():
                images_in_class = []
                for img_file in class_folder.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                        images_in_class.append(img_file)
                
                if images_in_class:
                    print(f"Found {len(images_in_class)} images in raw-img/{class_folder.name}")
                    for img_file in images_in_class:
                        image_paths.append((str(img_file), class_folder.name))
                    found_structure = True
    
    # Strategy 4: Recursive search if nothing found yet
    if not found_structure:
        print("No obvious structure found, doing recursive search...")
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    file_path = Path(root) / file
                    # Try to extract class name from parent directory
                    parent_dir = Path(root).name
                    if parent_dir != dataset_path.name:  # Don't use root dataset name
                        image_paths.append((str(file_path), parent_dir))
                        if not found_structure:
                            print(f"Found images with recursive search in: {parent_dir}")
                            found_structure = True
    
    return image_paths

def load_model(model_path, device):
    """Load the saved model"""
    # Load the saved data
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    num_classes = checkpoint['num_classes']
    model = CNN(num_classes=num_classes)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, checkpoint['class_names']

def predict_image(model, image_path, class_names, device, transform):
    """Predict the class of a single image"""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item()
    
    return predicted_label, confidence_score, probabilities[0]

def display_prediction(image_path, true_class, predicted_label, confidence, all_probabilities, class_names):
    """Display image with prediction information"""
    # Load and display image
    image = Image.open(image_path)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    
    # Create title with prediction info
    is_correct = "✓ CORRECT" if predicted_label == true_class else "✗ WRONG"
    title_color = 'green' if predicted_label == true_class else 'red'
    
    ax1.set_title(f'True: {true_class}\nPredicted: {predicted_label} ({confidence:.3f})\n{is_correct}', 
                 fontsize=14, color=title_color, fontweight='bold')
    
    # Create probability bar chart
    probabilities_cpu = all_probabilities.cpu().numpy()
    bars = ax2.barh(class_names, probabilities_cpu)
    
    # Color the predicted class bar
    predicted_idx = class_names.index(predicted_label)
    bars[predicted_idx].set_color('red' if predicted_label != true_class else 'green')
    
    # Highlight true class bar with border
    if true_class in class_names:
        true_idx = class_names.index(true_class)
        bars[true_idx].set_edgecolor('blue')
        bars[true_idx].set_linewidth(3)
    
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.set_xlim(0, 1)
    
    # Add probability values as text
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities_cpu)):
        ax2.text(prob + 0.01, i, f'{prob:.3f}', va='center')
    
    plt.tight_layout()
    return fig

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "trained_animal_classifier.pth"
    
    # Same transform as used in training
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    # Load the model
    print("Loading model...")
    model, class_names = load_model(model_path, device)
    print("Model loaded successfully!")
    print(f"Classes: {class_names}")
    print(f"Using device: {device}")
    
    # Get dataset path
    dataset_path = kagglehub.dataset_download("alessiocorrado99/animals10")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    # Explore dataset structure first
    explore_dataset_structure(dataset_path)
    
    # Get all images from dataset using flexible approach
    print("\nLoading image list...")
    all_images = get_all_images_from_dataset_flexible(dataset_path)
    
    if not all_images:
        print("No images found in dataset!")
        print("\nLet's do a manual check of the dataset structure:")
        
        # Manual exploration
        dataset_path_obj = Path(dataset_path)
        print(f"Dataset path exists: {dataset_path_obj.exists()}")
        print(f"Is directory: {dataset_path_obj.is_dir()}")
        
        if dataset_path_obj.exists():
            print("Contents of dataset directory:")
            for item in dataset_path_obj.iterdir():
                print(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        return
    
    print(f"Found {len(all_images)} images")
    
    # Show class distribution
    class_counts = {}
    for _, true_class in all_images:
        class_counts[true_class] = class_counts.get(true_class, 0) + 1
    
    print("\nClass distribution in dataset:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} images")
    
    print("\nStarting image display...")
    print("Instructions:")
    print("- Click anywhere on the image or press SPACE/ENTER to see next image")
    print("- Press 'q' to quit")
    print("- Close the window to exit")
    
    # Shuffle images for random order
    random.shuffle(all_images)
    
    # Statistics tracking
    correct_predictions = 0
    total_predictions = 0
    
    # Set up matplotlib for interactive use
    plt.ion()  # Turn on interactive mode
    
    # Global flag to control the loop
    quit_flag = False
    
    def on_key_press(event):
        nonlocal quit_flag
        if event.key == 'q':
            quit_flag = True
            plt.close('all')
    
    def on_mouse_click(event):
        # Just close the current figure, don't set quit flag
        plt.close(event.canvas.figure)
    
    try:
        for i, (image_path, true_class) in enumerate(all_images):
            if quit_flag:
                break
                
            try:
                # Make prediction
                predicted_label, confidence, all_probabilities = predict_image(
                    model, image_path, class_names, device, transform
                )
                
                # Update statistics
                total_predictions += 1
                if predicted_label == true_class:
                    correct_predictions += 1
                
                # Display the result
                fig = display_prediction(image_path, true_class, predicted_label, 
                                       confidence, all_probabilities, class_names)
                
                # Add accuracy to window title
                accuracy = correct_predictions / total_predictions * 100
                fig.suptitle(f'Image {total_predictions}/{len(all_images)} | Accuracy: {accuracy:.1f}%', 
                           fontsize=16, fontweight='bold')
                
                # Connect event handlers
                fig.canvas.mpl_connect('key_press_event', on_key_press)
                fig.canvas.mpl_connect('button_press_event', on_mouse_click)
                
                plt.show()
                
                # Print info to terminal
                print(f"\nImage {total_predictions}/{len(all_images)}: {os.path.basename(image_path)}")
                print(f"True: {true_class} | Predicted: {predicted_label} ({confidence:.3f})")
                print(f"Running accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
                print("Click image or press any key to continue, 'q' to quit...")
                
                # Wait for the figure to be closed
                while plt.fignum_exists(fig.number) and not quit_flag:
                    plt.pause(0.1)
                
                # If quit flag is set, break out of loop
                if quit_flag:
                    break
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        plt.close('all')
        
        # Final statistics
        if total_predictions > 0:
            final_accuracy = correct_predictions / total_predictions * 100
            print("\nFinal Results:")
            print(f"Total images tested: {total_predictions}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Final accuracy: {final_accuracy:.2f}%")
        
        print("Testing complete!")

if __name__ == "__main__":
    main()