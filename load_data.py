# Import necessary libraries
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from torch.utils.data import Dataset
import torch


# Initialize CLIP processor and tokenizer
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

# Load the dataset from Hugging Face
raw_dataset = load_dataset("nlphuji/flickr30k", split="test[:5000]", )

# Split the dataset into training and testing sets
train_test_split = raw_dataset.train_test_split(test_size=0.3)
train = train_test_split['train']
test = train_test_split['test']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        #print('IMG TENSOR SHAPE', img_tensor.shape) # channels, height, width
        
        # ---- Tokenize input caption ----
        caption = caption_list[0] # get the first caption in the list
        print('caption len:', len(caption))
        print('caption:', caption)
        tokens = self.tokenizer(caption, padding="max_length", max_length=18, return_tensors="pt", truncation=True)

        input_ids_full = tokens["input_ids"].to(self.device)  # [1, seq_len]
        #print('text_input_ids_full shape:', input_ids_full.shape)
        mask = tokens["attention_mask"].to(self.device) # get the mask out

        with torch.no_grad():
            # Use only embedding layer from CLIP
            text_embeddings = self.clip_model.text_model.embeddings(input_ids_full).squeeze(0).to(self.device)

            # Get the CLIP encoded image embeddings
            patch_embeddings = self.clip_model.vision_model(**img_tensor).last_hidden_state[:, 1:, :].squeeze(0).to(self.device) # shape: [1, num_patches, hidden_dim]
            #print('Patch embeddings shape:', patch_embeddings.shape)           
            
            
        target_ids = input_ids_full.squeeze(0).to(self.device)

        print('IMG EMBEDDINGS SHAPE', patch_embeddings.shape)
        print('TEXT EMBEDDINGS SHAPE', text_embeddings.shape)
        print('TARGET IDS SHAPE', target_ids.shape)
        print('MASK SHAPE', mask.shape)

        return patch_embeddings, text_embeddings, target_ids, mask
            

if __name__ == "__main__":
    # Process a single example from the test set
    test_caption = CaptionDataset(test)
    example = test_caption[0]
    
    print("\nProcessing completed successfully!")

