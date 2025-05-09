# Import necessary libraries
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from torch.utils.data import Dataset, DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset and create train and test sets
raw_dataset = load_dataset("nlphuji/flickr30k", split='test[:5000]')
train_test_split = raw_dataset.train_test_split(test_size=0.3, seed=42)
train = train_test_split['train']
test = train_test_split['test']

class CaptionDataset(Dataset):
    def __init__(self, dataset, clip_model_name="openai/clip-vit-base-patch32", device=device):
        self.image = dataset['image']
        self.caption_list = dataset['caption']
        
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).eval().to(self.device)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        caption_list = self.caption_list[idx]
        
        # ---- Encode image with CLIP ----
        img_tensor = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get first 3 captions (or fewer if not available)
        num_captions = min(3, len(caption_list))
        captions = caption_list[:num_captions]
        
        # Process each caption
        all_text_embeddings = []
        all_target_ids = []
        all_masks = []
        
        with torch.no_grad():
            # Get image embeddings once
            patch_embeddings = self.clip_model.vision_model(**img_tensor).last_hidden_state[:, 1:, :].squeeze(0).to(self.device)
            
            # Process each caption
            for caption in captions:
                # Tokenize caption
                tokens = self.tokenizer(caption, padding="max_length", max_length=18, return_tensors="pt", truncation=True)
                input_ids_full = tokens["input_ids"].to(self.device)
                mask = tokens["attention_mask"].to(self.device)
                
                # Get text embeddings
                text_embeddings = self.clip_model.text_model.embeddings(input_ids_full).squeeze(0).to(self.device)
                
                all_text_embeddings.append(text_embeddings)
                all_target_ids.append(input_ids_full.squeeze(0))
                all_masks.append(mask)
        
        # Stack all embeddings and targets
        text_embeddings = torch.stack(all_text_embeddings)
        target_ids = torch.stack(all_target_ids)
        masks = torch.stack(all_masks)
        
        # Duplicate image embeddings to match number of captions
        patch_embeddings = patch_embeddings.unsqueeze(0).repeat(num_captions, 1, 1)
        
        print('IMG EMBEDDINGS SHAPE', patch_embeddings.shape)
        print('TEXT EMBEDDINGS SHAPE', text_embeddings.shape)
        print('TARGET IDS SHAPE', target_ids.shape)
        print('MASK SHAPE', masks.shape)
        
        return patch_embeddings, text_embeddings, target_ids, masks
            

if __name__ == "__main__":
    # Create datasets
    train_dataset = CaptionDataset(train)
    test_dataset = CaptionDataset(test)
    
    # Test a single example
    example = train_dataset[0]
    patch_embeddings, text_embeddings, target_ids, masks = example
    
    print("\nProcessing completed successfully!")
    