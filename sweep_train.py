import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from decoder import Decoder
import random

def train(config=None):
    # Initialize wandb
    with wandb.init(config=config):
        # If called by wandb.agent, config will be passed
        config = wandb.config
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models and processors
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load dataset
        raw_dataset = load_dataset("nlphuji/flickr30k", split='test[:5000]')
        train_test_split = raw_dataset.train_test_split(test_size=0.3)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
        # Initialize decoder with sweep parameters
        decoder = Decoder(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            mlp_dimension=config.mlp_dimension,
            num_layers=config.num_layers,
            vocab_size=49408
        ).to(device)
        
        # Initialize optimizer with sweep learning rate
        optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(config.epochs):
            decoder.train()
            epoch_losses = []
            
            # Training phase
            train_pbar = tqdm(train_dataset, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
            
            for batch_idx, sample in enumerate(train_pbar):
                # Process image
                image = sample['image']
                caption = sample['caption'][0]
                
                # Process image through CLIP
                inputs = clip_processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    # Get image embeddings
                    patch_embeddings = clip_model.vision_model(**inputs).last_hidden_state[:, 1:, :].squeeze(0)
                    
                    # Project image features
                    image_projection_layer = nn.Linear(patch_embeddings.shape[-1], config.embedding_dim).to(device)
                    img_features = image_projection_layer(patch_embeddings)
                
                # Tokenize caption
                tokens = tokenizer(caption, padding="max_length", max_length=32, 
                                 return_tensors="pt", truncation=True)
                input_ids = tokens["input_ids"].to(device)
                mask = tokens["attention_mask"].to(device)
                
                # Get text embeddings
                with torch.no_grad():
                    text_embeddings = clip_model.text_model.embeddings(input_ids).squeeze(0)
                
                # Forward pass
                decoder_inputs = torch.cat([img_features.unsqueeze(0), text_embeddings.unsqueeze(0)], dim=1)
                logits = decoder(decoder_inputs)
                
                # Calculate loss
                text_logits = logits[:, -32:, :]
                text_targets = input_ids[:, -32:]
                text_mask = mask[:, -32:]
                
                # Reshape for loss calculation
                text_logits = text_logits.reshape(-1, text_logits.size(-1))
                text_targets = text_targets.reshape(-1)
                text_mask = text_mask.reshape(-1)
                
                # Calculate loss only on masked positions
                loss = criterion(
                    text_logits[text_mask.bool()],
                    text_targets[text_mask.bool()]
                )
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Store loss
                epoch_losses.append(loss.item())
                
                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Calculate average training loss
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            
            # Validation phase
            decoder.eval()
            val_losses = []
            
            with torch.no_grad():
                val_pbar = tqdm(test_dataset, desc=f'Epoch {epoch+1}/{config.epochs} [Val]')
                
                for batch_idx, sample in enumerate(val_pbar):
                    # Process image
                    image = sample['image']
                    caption = sample['caption'][0]
                    
                    # Process image through CLIP
                    inputs = clip_processor(images=image, return_tensors="pt").to(device)
                    
                    # Get image embeddings
                    patch_embeddings = clip_model.vision_model(**inputs).last_hidden_state[:, 1:, :].squeeze(0)
                    
                    # Project image features
                    image_projection_layer = nn.Linear(patch_embeddings.shape[-1], config.embedding_dim).to(device)
                    img_features = image_projection_layer(patch_embeddings)
                    
                    # Tokenize caption
                    tokens = tokenizer(caption, padding="max_length", max_length=32, 
                                     return_tensors="pt", truncation=True)
                    input_ids = tokens["input_ids"].to(device)
                    mask = tokens["attention_mask"].to(device)
                    
                    # Get text embeddings
                    text_embeddings = clip_model.text_model.embeddings(input_ids).squeeze(0)
                    
                    # Forward pass
                    decoder_inputs = torch.cat([img_features.unsqueeze(0), text_embeddings.unsqueeze(0)], dim=1)
                    logits = decoder(decoder_inputs)
                    
                    # Calculate loss
                    text_logits = logits[:, -32:, :]
                    text_targets = input_ids[:, -32:]
                    text_mask = mask[:, -32:]
                    
                    # Reshape for loss calculation
                    text_logits = text_logits.reshape(-1, text_logits.size(-1))
                    text_targets = text_targets.reshape(-1)
                    text_mask = text_mask.reshape(-1)
                    
                    # Calculate loss only on masked positions
                    val_loss = criterion(
                        text_logits[text_mask.bool()],
                        text_targets[text_mask.bool()]
                    )
                    
                    val_losses.append(val_loss.item())
                    val_pbar.set_postfix({'val_loss': f'{val_loss.item():.4f}'})
            
            # Calculate average validation loss
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            # Log metrics to wandb
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "epoch": epoch + 1
            })
            
            # Save model if it's the best so far
            if avg_val_loss < wandb.run.summary.get("best_val_loss", float('inf')):
                wandb.run.summary["best_val_loss"] = avg_val_loss
                torch.save(decoder.state_dict(), f"best_model_{wandb.run.id}.pt")
                wandb.save(f"best_model_{wandb.run.id}.pt")

def main():
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-5,
                'max': 1e-3,
                'distribution': 'log_uniform'
            },
            'embedding_dim': {
                'values': [256, 512, 768]
            },
            'num_heads': {
                'values': [4, 8, 12]
            },
            'mlp_dimension': {
                'values': [1024, 2048, 3072]
            },
            'num_layers': {
                'values': [2, 4, 6]
            },
            'epochs': {
                'value': 5
            }
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="image-captioning-sweeps")
    
    # Run sweep
    wandb.agent(sweep_id, function=train, count=10)  # Run 10 trials

if __name__ == "__main__":
    main() 