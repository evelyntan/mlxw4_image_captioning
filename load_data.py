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

class CaptionDataset(Dataset):
    def __init__(self, dataset, clip_model_name="openai/clip-vit-base-patch32"):
        self.image = dataset['image']
        self.caption_list = dataset['caption']

        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        # Keep CLIP model on CPU during initialization
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).eval()

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        caption_list = self.caption_list[idx]
        
        # ---- Encode image with CLIP ----
        # Keep tensors on CPU during processing
        img_tensor = self.processor(images=image, return_tensors="pt")
        
        # Get first 3 captions (or fewer if not available)
        num_captions = min(3, len(caption_list))
        captions = caption_list[:num_captions]
        
        # Process each caption
        all_text_embeddings = []
        all_target_ids = []
        all_masks = []
        
        with torch.no_grad():
            # Get image embeddings on CPU
            patch_embeddings = self.clip_model.vision_model(**img_tensor).last_hidden_state[:, 1:, :].squeeze(0)
            
            # Process each caption
            for caption in captions:
                # Tokenize caption
                tokens = self.tokenizer(caption, padding="max_length", max_length=18, return_tensors="pt", truncation=True)
                input_ids_full = tokens["input_ids"]
                mask = tokens["attention_mask"]
                
                # Get text embeddings on CPU
                text_embeddings = self.clip_model.text_model.embeddings(input_ids_full).squeeze(0)
                
                all_text_embeddings.append(text_embeddings)
                all_target_ids.append(input_ids_full.squeeze(0))
                all_masks.append(mask)
        
        # Stack all embeddings and targets on CPU
        text_embeddings = torch.stack(all_text_embeddings)  # [num_captions, seq_len, 512]
        target_ids = torch.stack(all_target_ids)  # [num_captions, seq_len]
        masks = torch.stack(all_masks)  # [num_captions, seq_len]
        
        print(f"After stacking:")
        print(f"text_embeddings: {text_embeddings.shape}")
        print(f"target_ids: {target_ids.shape}")
        print(f"masks: {masks.shape}")
        
        # Duplicate image embeddings to match number of captions
        patch_embeddings = patch_embeddings.unsqueeze(0).repeat(num_captions, 1, 1)  # [num_captions, 49, 768]
        print(f"After duplicating image embeddings: {patch_embeddings.shape}")
        
        # Reshape tensors to combine batch and caption dimensions
        batch_size = num_captions
        patch_embeddings = patch_embeddings.view(-1, 49, 768)  # [batch_size, num_patches, hidden_dim]
        text_embeddings = text_embeddings.view(-1, 18, 512)  # [batch_size, seq_len, hidden_dim]
        target_ids = target_ids.view(-1, 18)  # [batch_size, seq_len]
        masks = masks.view(-1, 18)  # [batch_size, seq_len]
        
        print(f"After final reshaping:")
        print(f"patch_embeddings: {patch_embeddings.shape}")
        print(f"text_embeddings: {text_embeddings.shape}")
        print(f"target_ids: {target_ids.shape}")
        print(f"masks: {masks.shape}")
        
        return patch_embeddings, text_embeddings, target_ids, masks
            

if __name__ == "__main__":
    # Create datasets
    train_dataset = CaptionDataset(train)
    test_dataset = CaptionDataset(test)
    
    # Test a single example
    example = train_dataset[0]
    patch_embeddings, text_embeddings, target_ids, masks = example
    
    print("\nProcessing completed successfully!")
    