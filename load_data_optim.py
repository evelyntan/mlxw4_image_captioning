# Import necessary libraries
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from torch.utils.data import Dataset
import torch


# load dataset and create train and test sets
raw_dataset = load_dataset("nlphuji/flickr30k", split='test[:5000]')
train_test_split = raw_dataset.train_test_split(test_size=0.3, seed=42)
train = train_test_split['train']
test = train_test_split['test']

clip_model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
tokenizer = CLIPTokenizer.from_pretrained(clip_model_name, use_fast=True)
clip_model = CLIPModel.from_pretrained(clip_model_name).eval()


class CaptionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['image'], item['caption']
    

def collate_fn(batch):
    # batch is a list of (image, caption_list) tuples
    images = [item[0] for item in batch]
    caption_lists = [item[1] for item in batch]
    
    # Process each image-caption pair
    all_patch_embeddings = []
    all_text_embeddings = []
    all_target_ids = []
    all_masks = []
    
    with torch.no_grad():
        for image, caption_list in zip(images, caption_lists):
            # ---- Encode image with CLIP ----
            img_tensor = processor(images=image, return_tensors="pt")
            patch_embeddings = clip_model.vision_model(**img_tensor).last_hidden_state[:, 1:, :].squeeze(0)
            
            # Get first 3 captions (or fewer if not available)
            num_captions = min(3, len(caption_list))
            captions = caption_list[:num_captions]
            
            # Process each caption
            for caption in captions:
                # Tokenize caption
                tokens = tokenizer(caption, padding="max_length", max_length=18, return_tensors="pt", truncation=True)
                input_ids_full = tokens["input_ids"]
                mask = tokens["attention_mask"]
                
                # Get text embeddings
                text_embeddings = clip_model.text_model.embeddings(input_ids_full).squeeze(0)
                
                all_patch_embeddings.append(patch_embeddings)
                all_text_embeddings.append(text_embeddings)
                all_target_ids.append(input_ids_full.squeeze(0))
                all_masks.append(mask)
    
    # Stack all batches
    patch_embeddings = torch.stack(all_patch_embeddings)  # [batch_size*num_captions, 49, 768]
    text_embeddings = torch.stack(all_text_embeddings)   # [batch_size*num_captions, 18, 512]
    target_ids = torch.stack(all_target_ids)            # [batch_size*num_captions, 18]
    masks = torch.stack(all_masks)                      # [batch_size*num_captions, 18]
    
    # Reshape to maintain batch and num_captions dimensions
    batch_size = len(images)
    num_captions = 3  # since we take max 3 captions per image
    
    patch_embeddings = patch_embeddings.view(batch_size, num_captions, 49, 768)  # [batch_size, num_captions, 49, 768]
    text_embeddings = text_embeddings.view(batch_size, num_captions, 18, 512)   # [batch_size, num_captions, 18, 512]
    target_ids = target_ids.view(batch_size, num_captions, 18)                  # [batch_size, num_captions, 18]
    masks = masks.view(batch_size, num_captions, 18)                           # [batch_size, num_captions, 18]
    
    return patch_embeddings, text_embeddings, target_ids, masks