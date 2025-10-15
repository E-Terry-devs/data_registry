import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import cv2
from functools import reduce
import argparse

def binary_image_mask_to_cvat_rle(image: np.ndarray) -> tuple:
    """Convert binary image mask to CVAT tight object RLE"""
    istrue = np.argwhere(image == 1).transpose()
    
    if istrue.size == 0:
        return "", (0, 0, 0, 0)
    
    top = int(istrue[0].min())
    left = int(istrue[1].min())
    bottom = int(istrue[0].max())
    right = int(istrue[1].max())
    roi_mask = image[top:bottom + 1, left:right + 1]

    # compute RLE values
    def reduce_fn(acc, v):
        if v == acc['val']:
            acc['res'][-1] += 1
        else:
            acc['val'] = v
            acc['res'].append(1)
        return acc
    
    roi_rle = reduce(
        reduce_fn,
        roi_mask.flat,
        {'res': [0], 'val': False}
    )['res']

    # Convert list to comma-separated string
    roi_rle_str = ', '.join(map(str, roi_rle))

    return roi_rle_str, (top, left, right - left + 1, bottom - top + 1)

def connected_components(binary_mask):
    """Get connected components from binary mask"""
    num_labels, labels_im = cv2.connectedComponents(binary_mask.astype(np.uint8))
    return num_labels, labels_im

def create_masks(num_labels, labels_im):
    """Create individual masks for each connected component"""
    masks = []
    for label in range(1, num_labels):  # start from 1 to exclude the background
        mask = (labels_im == label).astype(np.uint8)
        masks.append(mask)
    return masks

def parse_colored_mask(mask_path):
    """Parse colored mask and extract class information"""
    # Load mask
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Could not load mask: {mask_path}")
        return None
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    # Define your color mapping
    color_to_class = {
        (0, 0, 0): "soil",           # black
        (197, 11, 249): "crop",      # purple  
        (19, 156, 181): "weed"       # blueish
    }
    
    # Convert colored mask to class mask
    height, width = mask.shape[:2]
    class_mask = np.zeros((height, width), dtype=np.uint8)
    
    for color, class_name in color_to_class.items():
        # Find pixels matching this color (with some tolerance for compression artifacts)
        color_diff = np.sqrt(np.sum((mask - np.array(color))**2, axis=2))
        class_pixels = color_diff < 10  # tolerance for JPEG compression
        
        if class_name == "soil":
            class_mask[class_pixels] = 0
        elif class_name == "crop":
            class_mask[class_pixels] = 1
        elif class_name == "weed":
            class_mask[class_pixels] = 2
    
    return class_mask

def sam_masks_to_cvat(images_dir, masks_dir, output_xml):
    """
    Convert SAM fine-tuned masks to CVAT annotation XML format.
    
    Parameters:
    - images_dir (str): Directory containing original RGB images
    - masks_dir (str): Directory containing colored mask files (*_mask.png)
    - output_xml (str): Path to output CVAT XML file
    """
    
    # Create XML root
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"Found {len(image_files)} images to process")
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        
        # Find corresponding mask file
        base_name = os.path.splitext(image_file)[0]
        mask_file = f"{base_name}.png"
        mask_path = os.path.join(masks_dir, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {image_file}, skipping")
            continue
        
        print(f"Processing: {image_file}")
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load {image_file}, skipping")
            continue
            
        height, width = image.shape[:2]
        
        # Parse the colored mask
        class_mask = parse_colored_mask(mask_path)
        if class_mask is None:
            continue
        
        # Create image element in XML
        image_tag = ET.SubElement(root, "image", {
            "id": str(idx),
            "name": image_file,
            "width": str(width),
            "height": str(height)
        })
        
        # Process each class (crop=1, weed=2, skip soil=0)
        class_names = {1: "crop", 2: "weed"}
        
        for class_id, class_name in class_names.items():
            # Create binary mask for this class
            binary_mask = (class_mask == class_id).astype(np.uint8)
            
            if np.sum(binary_mask) == 0:
                continue  # Skip if no pixels of this class
            
            # Get connected components
            num_labels, labels_im = connected_components(binary_mask)
            
            if num_labels <= 1:
                continue  # No components found
            
            # Create individual masks for each component
            component_masks = create_masks(num_labels, labels_im)
            
            print(f"  Found {len(component_masks)} {class_name} components")
            
            # Create CVAT annotations for each component
            for component_idx, component_mask in enumerate(component_masks):
                # Skip very small components
                if np.sum(component_mask) < 20:
                    continue
                
                try:
                    # Convert to RLE format
                    rle, bbox = binary_image_mask_to_cvat_rle(component_mask)
                    
                    if rle == "":
                        continue
                    
                    top, left, bbox_width, bbox_height = bbox
                    
                    # Add mask annotation to XML
                    ET.SubElement(image_tag, "mask", {
                        "label": class_name,
                        "source": "semi-auto",
                        "occluded": "0",
                        "rle": rle,
                        "left": str(left),
                        "top": str(top),
                        "width": str(bbox_width),
                        "height": str(bbox_height),
                        "z_order": "0"
                    })
                    
                except Exception as e:
                    print(f"    Error processing {class_name} component {component_idx}: {e}")
                    continue
    
    # Save XML file
    tree = ET.ElementTree(root)
    
    try:
        with open(output_xml, 'wb') as f:
            tree.write(f, encoding='utf-8', xml_declaration=True)
        print(f"\n✅ CVAT annotation XML saved to: {output_xml}")
        print(f"You can now import this file into CVAT for refinement")
    except Exception as e:
        print(f"❌ Error saving CVAT annotation XML: {e}")

def create_labels_file(output_dir):
    """Create a labels.txt file for CVAT import"""
    labels_file = os.path.join(output_dir, "labels.txt")
    with open(labels_file, 'w') as f:
        f.write("crop\n")
        f.write("weed\n")
    print(f"Labels file created: {labels_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert SAM fine-tuned masks to CVAT format')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing original RGB images')
    parser.add_argument('--masks_dir', type=str, required=True,
                       help='Directory containing SAM generated colored masks (*_mask.png)')
    parser.add_argument('--output_xml', type=str, required=True,
                       help='Path to output CVAT XML file')
    parser.add_argument('--create_labels', action='store_true',
                       help='Create labels.txt file for CVAT')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.exists(args.images_dir):
        print(f"❌ Images directory not found: {args.images_dir}")
        return
    
    if not os.path.exists(args.masks_dir):
        print(f"❌ Masks directory not found: {args.masks_dir}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_xml)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert masks to CVAT format
    sam_masks_to_cvat(args.images_dir, args.masks_dir, args.output_xml)
    
    # Create labels file if requested
    if args.create_labels:
        create_labels_file(output_dir)
    
    print("\n📋 To import into CVAT:")
    print("1. Create a new task in CVAT")
    print("2. Upload your original images")
    print("3. Go to 'Import annotations'")
    print("4. Choose 'CVAT 1.1' format")
    print(f"5. Upload the generated XML file: {args.output_xml}")

if __name__ == "__main__":
    main()