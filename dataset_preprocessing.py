# Import necessary libraries
from datasets import load_dataset, DatasetDict
from transformers import CLIPProcessor, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# Load the dataset from Hugging Face
raw_datasets = load_dataset('your_dataset_name')

# Split the dataset into training and testing sets
train_test_split = raw_datasets['train'].train_test_split(test_size=0.3)
datasets = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Initialize CLIP processor and tokenizer
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

# Expand the dataset to have one entry per image-caption pair
expanded_train_data = []
expanded_test_data = []

def get_text_embeddings(caption):
    # Load CLIP tokenizer
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Freeze the weights
    clip_model.text_model.embeddings.token_embedding.weight.requires_grad = False

    # Tokenize the captions
    text_inputs = clip_tokenizer(text=caption, return_tensors="pt", padding=True, truncation=True)
    print("Vocab size:",  len(text_inputs['input_ids'][0]))
    
    # Extract the text embeddings
    token_embeddings = clip_model.text_model.embeddings.token_embedding

    # Get token IDs
    input_ids = text_inputs["input_ids"]

    # Get the embeddings to pass to decoder
    text_embeddings = token_embeddings(input_ids)

    print('Text embeddings shape:', text_embeddings.shape)

    return text_embeddings

# Modify the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, dataset_split):
        # Expand the dataset to have one entry per image-caption pair
        self.data = []
        for item in dataset_split:
            image = item['image']
            for caption in item['caption']:
                self.data.append({'image': image, 'caption': caption})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        caption = item['caption']

        # Preprocess the image
        image_tensor = processor(images=image, return_tensors='pt')['pixel_values'].squeeze()

        # Tokenize the caption
        text_inputs = tokenizer(caption, return_tensors='pt', padding=True, truncation=True)
        input_ids = text_inputs['input_ids']

        # Get the embeddings with positional encodings
        token_embeddings = clip_model.text_model.embeddings.token_embedding
        input_caption_embeddings = token_embeddings(input_ids)

        # Generate target caption
        target_caption = tokenizer(caption + tokenizer.eos_token, return_tensors='pt', padding=True, truncation=True)['input_ids'].squeeze()

        return image_tensor, input_caption_embeddings, target_caption

# Create dataset instances
train_dataset = CustomDataset(expanded_train_data)
test_dataset = CustomDataset(expanded_test_data)

# Define a collate function for batch processing

def collate_fn(batch):
    images, input_captions, target_captions = zip(*batch)
    images = torch.stack(images)
    input_captions = torch.nn.utils.rnn.pad_sequence(input_captions, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_captions = torch.nn.utils.rnn.pad_sequence(target_captions, batch_first=True, padding_value=tokenizer.pad_token_id)
    return images, input_captions, target_captions

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# Save the preprocessed datasets to local files

torch.save(train_dataset, 'train_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')

# To load the datasets in the future, use:
# train_dataset = torch.load('train_dataset.pt')
# test_dataset = torch.load('test_dataset.pt') 