
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPImageProcessor
import torch
import os
import json
from torch.utils.data import DataLoader, Dataset
from src.data_utils import FashionIQDataset
from tqdm import tqdm


def collate_fn_filter_none(batch):
    """
    Custom collate function to filter out None values (failed image loads).
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def get_image_tensor(
    dataloader: DataLoader,
    save_flod: str,
    clip_image_encoder: CLIPVisionModelWithProjection,
):
    """
    处理数据集的图片, 保存的tensor
    """
    image_tensor = []
    for batch in tqdm(dataloader, desc="Processing images"):
        if batch is None: # Skip empty batches
            continue
        
        images = batch[1]['pixel_values'][0]
        
        # Move images to the same device as the model
        images = images.to(clip_image_encoder.device)

        with torch.no_grad():
            image_features = clip_image_encoder(images).image_embeds
        image_tensor.append(image_features.cpu())
    
    if not image_tensor:
        print("No images processed.")
        return

    image_tensor = torch.cat(image_tensor, dim=0)
    print(f"Image tensor shape: {image_tensor.shape}")
    os.makedirs(save_flod, exist_ok=True)
    save_path = os.path.join(save_flod, 'image_tensor.pt')
    torch.save(image_tensor, save_path)
    print(f"Saved image tensors to {save_path}")






    return None


if __name__ == "__main__":
    print('CLIP preprocess pipeline is used')
    save_flod = "fashionIQ_dataset/image_vlm_feature/dress"
    preprocess = CLIPImageProcessor(
        crop_size={'height': 224, 'width': 224},
        do_center_crop=True,
        do_convert_rgb=True,
        do_normalize=True,
        do_rescale=True,
        do_resize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        resample=3,
        size={'shortest_edge': 224},
    )
    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

    dataset = FashionIQDataset(
        split='train',
        dress_types=['dress'],
        mode='classic',
        preprocess=preprocess,
    )


    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn_filter_none # Use the custom collate function
    )
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For Mac M1/M2 chips
    if torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    clip_image_encoder = clip_image_encoder.to(device)

    get_image_tensor(
        dataloader,
        save_flod=save_flod,
        clip_image_encoder=clip_image_encoder,
    )
    print('Done')   