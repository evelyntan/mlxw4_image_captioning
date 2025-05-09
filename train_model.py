import torch.optim as optim
from tqdm import tqdm
from transformers import CLIPTokenizer
from decoder import Decoder
import torch
import torch.nn as nn

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters for decoder
embedding_dim = 256
num_heads = 8
mlp_dimension = 2048
num_layers = 4
input_sequence_length = 32
vocab_size = 49408

decoder = Decoder(embedding_dim=embedding_dim, num_heads=num_heads, 
                  mlp_dimension=mlp_dimension, num_layers=num_layers, 
                  vocab_size=vocab_size).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

# Move projection layer to device
image_projection_layer = nn.Linear(768, embedding_dim).to(device)
text_projection_layer = nn.Linear(512, embedding_dim).to(device)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):

    decoder.train()

    # Training phase
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', mininterval=0.1)

    epoch_losses = [] # store all individual losses and calculate the true average at the end

    # Iterate over the DataLoader and print the outputs
    for batch_idx, (patch_embeddings, text_embeddings, target_ids, mask) in enumerate(train_pbar):

        # Move everything to device
        patch_embeddings = patch_embeddings.to(device)
        text_embeddings = text_embeddings.to(device)
        mask = mask.to(device)
        targets = target_ids.to(device)
        img_features = image_projection_layer(patch_embeddings)
        text_features = text_projection_layer(text_embeddings)

    
        #print('\nProjected img shape:', img_features.shape)
        #print('Projected text shape:', text_embeddings.shape)

        #print('Mask shape:', mask.shape)


        text_embeddings = text_embeddings.to(device)
        img_features = img_features.to(device)
        targets = target_ids.to(device)

        # Forward pass through the decoder
        logits = decoder(text_features, img_features)

        #print('logits shape:', logits.shape)

        # Compute loss
        # Shift logits and targets to align for cross-entropy
        logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
        targets = target_ids[:, 1:].contiguous().view(-1)
        loss = criterion(logits, targets)

        # Calculate loss only on text portion
        #text_logits = logits[:, -32:, :] # Shape: [batch_size, 32, vocab_size]
        #print('text_logits shape:', text_logits.shape)

        # Sqeuueze mask to remove th eextra dimension
        #mask=mask.squeeze(1)
        #print('mask shape:', mask.shape)

        # Apply teacher forcing to text portion only 
        #text_logits = text_logits[:, :-1].contiguous().view(-1, text_logits.size(-1))
        #text_targets = target_ids[:, 1:].contiguous().view(-1)
        #text_mask = mask[:, 1:].contiguous().view(-1)

        #print('text_logits shape:', text_logits.shape)
        #print('text_targets shape:', text_targets.shape)
       
        # Calculate loss only on masked positions
        #text_mask = text_mask.bool()
        #loss = criterion(
        #    text_logits[text_mask],
        #    text_targets[text_mask]
        #)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Store the loss for this batch
        epoch_losses.append(loss.item())
        
        # Update progress bar with current batch loss
        train_pbar.set_postfix({
            'batch_loss': f'{loss.item():.4f}'
        })
        train_pbar.update(1)

    # Close progress bar for this epoch
    train_pbar.close()
    
    # Print epoch summary
    # Calculate and print training epoch average loss
    avg_train_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
    print(" ")

     # Validation phase
    decoder.eval()  # Set model to evaluation mode
    val_losses = []  # Store all validation losses
    
    with torch.no_grad():  # Disable gradient calculation for validation
        val_pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', mininterval=0.1)
        
        for batch_idx, (patch_embeddings, text_embeddings, target_ids, mask) in enumerate(val_pbar):

            # Move everything to device
            patch_embeddings = patch_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)
            mask = mask.to(device)
            targets = target_ids.to(device)
            img_features = image_projection_layer(patch_embeddings)
            text_features = text_projection_layer(text_embeddings)


            # Forward pass through the decoder
            logits = decoder(text_features, img_features)

            # Compute loss
            # Shift logits and targets to align for cross-entropy
            logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
            targets = target_ids[:, 1:].contiguous().view(-1)
            val_loss = criterion(logits, targets)

            # Calculate loss only on text portion
            #text_logits = logits[:, -32:, :] # Shape: [batch_size, 32, vocab_size]
            #print('text_logits shape:', text_logits.shape)

            # Sqeuueze mask to remove th eextra dimension
            #mask=mask.squeeze(1)
            #print('mask shape:', mask.shape)

            # Apply teacher forcing to text portion only 
            #text_logits = text_logits[:, :-1].contiguous().view(-1, text_logits.size(-1))
            #text_targets = target_ids[:, 1:].contiguous().view(-1)
            #text_mask = mask[:, 1:].contiguous().view(-1)

            #print('text_logits shape:', text_logits.shape)
            #print('text_targets shape:', text_targets.shape)
        

            # Calculate loss only on masked positions
            #text_mask = text_mask.bool()
            #val_loss = criterion(
            #    text_logits[text_mask],
            #    text_targets[text_mask]
            #)

            # Store validation loss
            val_losses.append(val_loss.item())
            
            # Update progress bar with current validation loss
            val_pbar.set_postfix({
                'val_loss': f'{val_loss.item():.4f}'
            })
            val_pbar.update(1)
    
    # Close validation progress bar
    val_pbar.close()
    
    # Calculate average validation loss
    avg_val_loss = sum(val_losses) / len(val_losses)
    
    # Print epoch summary
    print(f"Epoch {epoch+1}/{num_epochs} Val Loss: {avg_val_loss:.4f}")

        
print("Training complete.")

