from model.rqkmeans import RQKmeans
from dataclasses import dataclass, field
import torch
import os
@dataclass
class ModelCfg:
    num_book: int = 0
    num_cluster: list = field(default_factory=list)
    dim: int = 512


@dataclass
class Cfg:

    model: ModelCfg = ModelCfg()
    seed: int = 42
    device: str = "cpu"

if __name__ == "__main__":
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from PIL import Image

    model_cfg = ModelCfg(
        num_book=10,
        num_cluster=[256, 256, 256, 256, 256, 256, 256, 256, 256, 256],
        dim=512,
    )
    cfg = Cfg(model=model_cfg, seed=42, device="cpu")
    os.makedirs('./model/checkpoints/tokenize/', exist_ok=True) 
    data = torch.load('fashionIQ_dataset/image_vlm_feature/dress/image_tensor.pt')
    tokenize = RQKmeans(cfg=cfg)
    tokenize.fit(data)
    torch.save(tokenize.state_dict(), './model/checkpoints/tokenize/rqkmeans.pth')

    # tokenize.load_state_dict(torch.load('./model/checkpoints/tokenize/rqkmeans.pth'))
    # tokenize.eval()
    # preprocess = CLIPImageProcessor(
    #     crop_size={'height': 224, 'width': 224},
    #     do_center_crop=True,
    #     do_convert_rgb=True,
    #     do_normalize=True,
    #     do_rescale=True,
    #     do_resize=True,
    #     image_mean=[0.48145466, 0.4578275, 0.40821073],
    #     image_std=[0.26862954, 0.26130258, 0.27577711],
    #     resample=3,
    #     size={'shortest_edge': 224},
    # )
    # clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    # image_1_path = 'fashionIQ_dataset/images/B00A0IEA3K.jpg'
    # image_2_path = 'fashionIQ_dataset/images/B00A0IE8IC.jpg'
    # image_1_emb = clip_image_encoder(torch.tensor(preprocess(Image.open(image_1_path))['pixel_values'][0]).unsqueeze(0)).image_embeds
    # image_2_emb = clip_image_encoder(torch.tensor(preprocess(Image.open(image_2_path))['pixel_values'][0]).unsqueeze(0)).image_embeds
    # sid1, res1 = tokenize(image_1_emb)
    # sid2, res2 = tokenize(image_2_emb)
    # print("Image 1 RQKmeans codes:", sid1)
    # print("Image 2 RQKmeans codes:", sid2)
