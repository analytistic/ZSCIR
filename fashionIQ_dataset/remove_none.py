import json
import os

def clear_none(image_idx_json, cap_idx_json):
    with open(image_idx_json, 'r') as f:
        image_idx_list = json.load(f)

    with open(cap_idx_json, 'r') as f:
        cap_idx_list = json.load(f)
    
    print(f"Initial number of images: {len(image_idx_list)}")
    print(f"Initial number of captions: {len(cap_idx_list)}")
    image_idx_list_clean = []
    cap_idx_list_clean = []
    for image_idx in image_idx_list:
        
        image_path = os.path.join('fashionIQ_dataset/images/', f"{image_idx}.jpg")
        if not os.path.exists(image_path):
            continue
        image_idx_list_clean.append(image_idx)

    for cap_idx in cap_idx_list:
        target = cap_idx['target']
        candidate = cap_idx['candidate']
        target_path = os.path.join('fashionIQ_dataset/images/', f"{target}.jpg")
        candidate_path = os.path.join('fashionIQ_dataset/images/', f"{candidate}.jpg")
        if not os.path.exists(target_path) or not os.path.exists(candidate_path):
            continue
        cap_idx_list_clean.append(cap_idx)

    # save
    with open(image_idx_json, 'w') as f:
        json.dump(image_idx_list_clean, f)
    with open(cap_idx_json, 'w') as f:
        json.dump(cap_idx_list_clean, f)

if __name__ == "__main__":
    clear_none('fashionIQ_dataset/image_splits/split.dress.test.json',
                'fashionIQ_dataset/captions/cap.dress.test.json')
    

    

        