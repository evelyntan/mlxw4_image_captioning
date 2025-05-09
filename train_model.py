import torch.optim as optim
from tqdm import tqdm
from transformers import CLIPTokenizer
from decoder import Decoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from load_data import CaptionDataset
import wandb
from datetime import datetime
import multiprocessing
import os

# Set multiprocessing start method to 'spawn' to use CUDA with multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def cleanup():
    # Cleanup CUDA resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train(config=None):
    # Get current timestamp for run name
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_number = config.run_number if hasattr(config, 'run_number') else 1
    run_name = f"run_{run_number}_{timestamp}"
    
    # Initialize wandb with custom run name
    with wandb.init(config=config, name=run_name):
        # If called by wandb agent, config will be set
        config = wandb.config
        
        # Set device for remote GPU
        if torch.cuda.is_available():
            # Set the default CUDA device
            torch.cuda.set_device(0)  # Use first GPU
            device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        # Load dataset and create train/test split
        raw_dataset = load_dataset("nlphuji/flickr30k", split='test[:5000]')
        train_test_split = raw_dataset.train_test_split(test_size=0.3)
        
        # Create datasets
        train_dataset = CaptionDataset(train_test_split['train'])
        test_dataset = CaptionDataset(train_test_split['test'])
        
        # Create dataloaders with persistent workers
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=1,  # Reduced from 2 to 1
            persistent_workers=False,  # Disabled persistent workers
            pin_memory=True,  # Enable pin memory for faster CPU->GPU transfer
            multiprocessing_context='spawn'  # Explicitly set spawn context
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=1,  # Reduced from 2 to 1
            persistent_workers=False,  # Disabled persistent workers
            pin_memory=True,  # Enable pin memory for faster CPU->GPU transfer
            multiprocessing_context='spawn'  # Explicitly set spawn context
        )
        
        # Initialize model
        decoder = Decoder(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            mlp_dimension=config.mlp_dimension,
            num_layers=config.num_layers,
            vocab_size=49408  # CLIP's vocab size
        ).to(device)
        
        # Initialize optimizer and loss
        optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Training loop
        for epoch in range(config.num_epochs):
            # Training phase
            decoder.train()
            train_losses = []
            
            train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Train]', mininterval=2)
            for batch_idx, (patch_embeddings, text_embeddings, target_ids, mask) in enumerate(train_pbar):
                # Move to device
                patch_embeddings = patch_embeddings.to(device)
                text_embeddings = text_embeddings.to(device)
                target_ids = target_ids.to(device)
                mask = mask.to(device)
                
                # Forward pass
                logits = decoder(text_embeddings, patch_embeddings)
                
                # Compute loss
                logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
                targets = target_ids.view(-1, target_ids.size(-1))[:, 1:].contiguous().view(-1)
                loss = criterion(logits, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Store loss
                train_losses.append(loss.item())
                
                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Calculate average training loss
            avg_train_loss = sum(train_losses) / len(train_losses)
            
            # Validation phase
            decoder.eval()
            val_losses = []
            
            with torch.no_grad():
                val_pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Val]', mininterval=2)
                for batch_idx, (patch_embeddings, text_embeddings, target_ids, mask) in enumerate(val_pbar):
                    # Move to device
                    patch_embeddings = patch_embeddings.to(device)
                    text_embeddings = text_embeddings.to(device)
                    target_ids = target_ids.to(device)
                    mask = mask.to(device)
                    
                    # Forward pass
                    logits = decoder(text_embeddings, patch_embeddings)
                    
                    # Compute loss
                    logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
                    targets = target_ids.view(-1, target_ids.size(-1))[:, 1:].contiguous().view(-1)
                    val_loss = criterion(logits, targets)
                    
                    # Store loss
                    val_losses.append(val_loss.item())
                    
                    # Update progress bar
                    val_pbar.set_postfix({'val_loss': f'{val_loss.item():.4f}'})
            
            # Calculate average validation loss
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            # Log to wandb
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "epoch": epoch
            })
            
            # Save model checkpoint with timestamp
            if (epoch + 1) % config.save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }
                checkpoint_name = f'checkpoint_{run_name}_epoch_{epoch+1}.pt'
                torch.save(checkpoint, checkpoint_name)
                wandb.save(checkpoint_name)
        
        # Save final model with timestamp
        final_model_name = f'model_{run_name}.pt'
        torch.save(decoder.state_dict(), final_model_name)
        wandb.save(final_model_name)

if __name__ == "__main__":
    # Initialize sweep
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'batch_size': {
                'values': [32, 64]
            },
            'num_epochs': {
                'values': [10, 15, 20]
            },
            'embedding_dim': {
                'values': [256, 512]
            },
            'num_heads': {
                'values': [8, 16]
            },
            'mlp_dimension': {
                'value': 2048
            },
            'num_layers': {
                'values': [4, 6]
            },
            'save_interval': {
                'value': 5
            }
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="image-captioning")
    
    # Run the sweep
    wandb.agent(sweep_id, function=train, count=10)
    print("Sweep completed!")

