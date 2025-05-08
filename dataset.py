import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel

class CaptionDataset(Dataset):
    def __init__(self, dataset, clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.image = dataset['image']
        self.caption_list = dataset['caption']
        
        self.device = device
        self.embedding_dim = 512  # Define the target embedding dimension

        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).eval().to(self.device)
        
        # Initialize the projection layer
        self.image_projection = nn.Linear(768, self.embedding_dim).to(self.device)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        caption_list = self.caption_list[idx]
        
        # ---- Encode image with CLIP ----
        img_tensor = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # ---- Tokenize input caption ----
        caption = caption_list[0]  # get the first caption in the list
        tokens = self.tokenizer(caption, padding="max_length", max_length=32, return_tensors="pt", truncation=True)

        input_ids_full = tokens["input_ids"].to(self.device)  # [1, seq_len]
        mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            # Use only embedding layer from CLIP
            text_embeddings = self.clip_model.text_model.embeddings(input_ids_full).squeeze(0).to(self.device)

            # Get the CLIP encoded image embeddings and project them
            patch_embeddings = self.clip_model.vision_model(**img_tensor).last_hidden_state[:, 1:, :].squeeze(0).to(self.device)
            patch_embeddings = self.image_projection(patch_embeddings)  # Project to match text embedding dimension
            
            # Get the first element of the projected patch embeddings
            first_patch_embedding = patch_embeddings[0]  # Shape: [512]
            
        target_ids = input_ids_full.squeeze(0).to(self.device)

        return first_patch_embedding, text_embeddings, target_ids, mask 