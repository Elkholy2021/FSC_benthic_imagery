import os
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from FSCBI.utils.gpu_utils import set_seed
import re
import random
from collections import Counter
from collections import defaultdict
import random
from collections import Counter
def to_underscore_words(text: str) -> str:
    # Remove leading/trailing spaces
    text = text.strip()
    
    # Replace any sequence of non-letter/non-number characters with "_"
    text = re.sub(r'[^A-Za-z0-9]+', '_', text)
    
    # Remove possible leading/trailing underscores
    text = text.strip('_')
    
    return text

def prepare_cropped_objects(
    images_root,
    output_root,
    data_json=None,
    overwrite=False,
    keep_original_background=False,
    bbox_expand_ratio=0.0,
    seed=42,
    draw_red_outline=False,
    debug=False,
    red_thickness=2
):


    set_seed(seed)
    os.makedirs(output_root, exist_ok=True)

    # ---------- Load JSON ----------
    data = data_json

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    # Maps
    image_id_to_info = {img["id"]: img for img in images}
    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in categories}

    # Cache images
    image_cache = {}

    metadata = []
    CC=0
    DD = 0
    for ann in tqdm(annotations, desc="Processing annotations"):
        ann_id = ann["id"]
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        category_name = cat_id_to_name.get(category_id, str(category_id))

        img_info = image_id_to_info.get(image_id)
        if img_info is None:
            continue

        image_file = img_info["file_name"]
        img_path = os.path.join(images_root, image_file)
        category_name = to_underscore_words(category_name)

        # Output path
        cat_dir = os.path.join(output_root, str(category_name))
        os.makedirs(cat_dir, exist_ok=True)
        crop_filename = f"obj_{ann_id}.png"
        crop_path = os.path.join(cat_dir, crop_filename)

        # if (not overwrite) and os.path.exists(crop_path):
        #     metadata.append({
        #         "object_id": ann_id,
        #         "image_id": image_id,
        #         "crop_path": crop_path,
        #         "category_id": category_id,
        #         "category_name": category_name,
        #         "image_file": image_file,
        #     })
        #     continue
            

        if not overwrite:
            if os.path.exists(crop_path):
                # Only include already-existing crops
                metadata.append({
                    "object_id": ann_id,
                    "image_id": image_id,
                    "crop_path": crop_path,
                    "category_id": category_id,
                    "category_name": category_name,
                    "image_file": image_file,
                })

            # Whether it exists or not, do not create/reprocess anything
            continue

        # ---------- Load image ----------
        if image_id in image_cache:
            img_rgb = image_cache[image_id]
        else:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image_cache[image_id] = img_rgb

        h, w, _ = img_rgb.shape

        # ---------- Original bbox ----------
        x, y, bw, bh = ann.get("bbox", [0, 0, w, h])

        # ---------- Expand bbox ----------
        if bbox_expand_ratio > 0:
            expand_w = bw * bbox_expand_ratio
            expand_h = bh * bbox_expand_ratio

            x -= expand_w
            y -= expand_h
            bw += 2 * expand_w
            bh += 2 * expand_h

        # Convert to final integers and keep inside image
        x0 = int(max(0, np.floor(x)))
        y0 = int(max(0, np.floor(y)))
        x1 = int(min(w, np.ceil(x + bw)))
        y1 = int(min(h, np.ceil(y + bh)))
        if debug:
        
            
            if len(ann.get('segmentation', [])[0]) <=8:
                CC+=1
                print(f"\033[92mNumber of points in bbox: {len(ann.get('segmentation', [])[0])}\033[0m")  # Debug line
        # safety fallback
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, w, h

        # ---------- Return only expanded bbox of original ---------
        if keep_original_background:
            crop = img_rgb[y0:y1, x0:x1, :]

        else:
            # ---------- Build mask ----------
            mask = np.zeros((h, w), dtype=np.uint8)
            seg = ann.get("segmentation", [])

            for poly in seg:
                if len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)

            if mask.sum() == 0:
                continue

            # ---------- Apply white background ----------
            white_bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
            obj_only = white_bg.copy()
            obj_only[mask == 1] = img_rgb[mask == 1]

            # Crop to expanded bbox
            crop = obj_only[y0:y1, x0:x1, :]




        
        if draw_red_outline:
            mask = np.zeros((h, w), dtype=np.uint8)
            seg = ann.get("segmentation", [])

            for poly in seg:
                if len(poly) <= 8:
                    DD +=1
                    continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.polylines(mask, [pts], isClosed=True, color=1, thickness=red_thickness)

            # Draw red outline on original image and crop to expanded bbox
            crop = img_rgb[y0:y1, x0:x1, :].copy()
            # Adjust mask coordinates to cropped region
            mask_crop = mask[y0:y1, x0:x1]
            crop[mask_crop == 1] = [255, 0, 0]  # Red color in RGB
        else:
            crop = crop
  





        # ---------- Save image ----------
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(crop_path, crop_bgr)

        # ---------- Record metadata ----------
        metadata.append({
            "object_id": ann_id,
            "image_id": image_id,
            "crop_path": crop_path,
            "category_id": category_id,
            "category_name": category_name,
            "image_file": image_file,
        })
    print(f"Number of annotations with less than 4 points: {CC}")
    print(f"\033[94mNumber of annotations with less than 4 points for red outline: {DD}\033[0m")  # Debug line

    return metadata


class CroppedObjectsDataset(Dataset):
    """
    PyTorch Dataset for the cropped object images generated by prepare_cropped_objects().

    Each item is a dict with:
        {
            "image": PIL.Image or transformed tensor,
            "object_id": int,
            "image_id": int,
            "crop_path": str,
            "category_id": int,
            "category_name": str,
            "image_file": str,  # original file name
        }
    """
    def __init__(self, metadata, transform=None, augment=False, seed=42):
        """
        Parameters
        ----------
        metadata : list of dict
            Output from prepare_cropped_objects().
        transform : callable or None
            Optional transform applied to the image (e.g. torchvision transforms).
        augment : bool, optional
            If True, apply random flipping and rotation augmentations (default: False).
        """
        self.metadata = metadata
        self.transform = transform
        self.augment = augment
        set_seed(seed)
        
        if self.augment:
            self.augmentation = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=180),
            ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item_meta = self.metadata[idx]
        img = Image.open(item_meta["crop_path"]).convert("RGB")
        # print(f"type(img): {type(img)}")
        if self.augment:
            img = self.augmentation(img)

        if self.transform is not None:
            img = self.transform(img)

        # Return everything you asked for:
        return {
            "image": img,
            "object_id": item_meta["object_id"],
            "image_id": item_meta["image_id"],
            "crop_path": item_meta["crop_path"],
            "category_id": item_meta["category_id"],
            "category_name": item_meta["category_name"],
            "image_file": item_meta["image_file"],
        }


def get_json_files(directory):
    train_json = None
    val_json = None
    test_json = None
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.abspath(os.path.join(directory, filename))
            basename = os.path.basename(filename).lower()
            
            if 'train' in basename:
                train_json = filepath
            elif 'val' in basename:
                val_json = filepath
            elif 'test' in basename:
                test_json = filepath
            elif 'all' in basename:
                all_json = filepath
    
    # return train_json, val_json, test_json , all_json
    return all_json


def get_num_categories(data_json):

    categories = data_json.get("categories", [])
    return len(categories)


def _split_50_50_by_image(metadata, seed=42):
    """
    50/50 split in which all objects of an image go to the same side (no image appears in both splits).
    Images are visited in a seeded random order and each one is assigned to the split that is further
    below its 50% target for the classes the image contains, which keeps the split roughly stratified.
    """
    rng = random.Random(seed)
    by_image = defaultdict(list)
    for item in metadata:
        by_image[item["image_id"]].append(item)
    image_ids = sorted(by_image.keys(), key=lambda x: str(x))
    rng.shuffle(image_ids)

    target = Counter(item["category_id"] for item in metadata)
    target = {c: n / 2.0 for c, n in target.items()}
    counts = [Counter(), Counter()]
    splits = [[], []]
    for img_id in image_ids:
        items = by_image[img_id]
        img_counts = Counter(item["category_id"] for item in items)
        # deficit of each split for this image's classes (positive = split still needs these classes)
        deficits = [sum((target[c] - counts[s][c]) * n for c, n in img_counts.items()) for s in (0, 1)]
        if deficits[0] > deficits[1]:
            s = 0
        elif deficits[1] > deficits[0]:
            s = 1
        else:
            s = rng.randint(0, 1)
        splits[s].extend(items)
        counts[s].update(img_counts)
    return splits[0], splits[1]


def split_50_50(dataset, seed=42, by="object"):
    """
    by="object": each class's objects are shuffled and halved (objects of one image can land on both sides).
    by="image":  all objects of an image stay on the same side (see _split_50_50_by_image).
    """
    set_seed(seed)
    random.seed(seed)
    if by == "image":
        split1_metadata, split2_metadata = _split_50_50_by_image(dataset.metadata, seed=seed)
        dataset1 = CroppedObjectsDataset(split1_metadata, transform=dataset.transform, augment=dataset.augment, seed=seed)
        dataset2 = CroppedObjectsDataset(split2_metadata, transform=dataset.transform, augment=dataset.augment, seed=seed)
        return dataset1, dataset2
    if by != "object":
        raise ValueError(f"Unknown split unit: {by} (use 'object' or 'image')")
    # Group metadata by category_id
    category_groups = defaultdict(list)
    for item in dataset.metadata:
        category_groups[item["category_id"]].append(item)
    
    # Split each category 50-50
    split1_metadata = []
    split2_metadata = []
    
    for category_id, items in category_groups.items():
        # Shuffle items within category for randomness
        shuffled_items = items.copy()
        random.shuffle(shuffled_items)
        
        # Split in half
        mid_point = len(shuffled_items) // 2
        split1_metadata.extend(shuffled_items[:mid_point])
        split2_metadata.extend(shuffled_items[mid_point:])
    
    # Create new dataset instances with same transform/augment settings
    dataset1 = CroppedObjectsDataset(
        split1_metadata, 
        transform=dataset.transform,
        augment=dataset.augment,
        seed=seed
    )
    dataset2 = CroppedObjectsDataset(
        split2_metadata,
        transform=dataset.transform,
        augment=dataset.augment,
        seed=seed
    )
    
    return dataset1, dataset2

def plot_label_distribution_comparison(dataset1, dataset2, title1="Dataset 1", title2="Dataset 2"):
    """Plot label distribution comparison for two datasets in a 1x2 subplot."""
    import matplotlib.pyplot as plt
    
    # Count labels in each dataset
    labels1 = [item["category_id"] for item in dataset1.metadata]
    labels2 = [item["category_id"] for item in dataset2.metadata]
    
    counter1 = Counter(labels1)
    counter2 = Counter(labels2)
    
    # Get all unique categories (sorted)
    all_categories = sorted(set(counter1.keys()) | set(counter2.keys()))
    
    # Prepare data for plotting
    counts1 = [counter1.get(cat, 0) for cat in all_categories]
    counts2 = [counter2.get(cat, 0) for cat in all_categories]
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot dataset 1
    ax1.bar(all_categories, counts1, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Category ID')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title1}\n(Total: {len(labels1)} samples)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot dataset 2
    ax2.bar(all_categories, counts2, color='coral', alpha=0.7)
    ax2.set_xlabel('Category ID')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{title2}\n(Total: {len(labels2)} samples)')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig
