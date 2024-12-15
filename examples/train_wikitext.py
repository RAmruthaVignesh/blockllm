"""
Example script demonstrating BlockLLM optimizer usage with IMDB dataset.
This script shows how to train a small language model using BlockLLM's sparse parameter updates.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from blockllm_torch.blockllm import BlockLLM, BlockLLMConfig
from torch.utils.data import DataLoader
from datasets import load_dataset

def prepare_data(tokenizer, max_length=256):  # Increased max_length for movie reviews
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    
    def tokenize_function(examples):
        # Format the text with sentiment
        texts = [
            f"Review: {text}\nSentiment: {'positive' if label == 1 else 'negative'}"
            for text, label in zip(examples['text'], examples['label'])
        ]
        
        # Filter out empty texts
        texts = [t for t in texts if t.strip()]
        if not texts:
            return {'input_ids': [], 'attention_mask': [], 'labels': []}
            
        # Tokenize the texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels for language modeling
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    # Tokenize and format dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        batch_size=100
    )
    
    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: len(example['input_ids']) > 0
    )
    
    return tokenized_dataset

def collate_fn(batch):
    # Convert lists to tensors
    return {
        'input_ids': torch.cat([torch.tensor(item['input_ids']) for item in batch]),
        'attention_mask': torch.cat([torch.tensor(item['attention_mask']) for item in batch]),
        'labels': torch.cat([torch.tensor(item['labels']) for item in batch])
    }

def train(model, train_dataloader, optimizer, num_epochs=3, device='cuda', scheduler=None):
    model.train()
    print(f"Training on device {device}")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            # optimizer.set_loss(loss.item())  # Set the loss value in the optimizer
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

            # Print progress
            if batch_idx % 10 == 0:

                # Training accuracy
                with torch.no_grad():
                    outputs = model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels)
                    accuracy = (outputs.logits.argmax(dim=-1) == labels).float().mean()
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
        # Compute average loss    
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Generate a sample completion after each epoch
        if batch_idx % 500 == 0:
            model.eval()
            with torch.no_grad():
                sample_text = "Review: This movie was absolutely"
                inputs = tokenizer(sample_text, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\nSample generation:\n{generated_text}\n")
            model.train()

def main():
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    num_epochs = 10

    # Seed for reproducibility
    torch.manual_seed(42)

    # Load model and tokenizer
    model_name = "gpt2"  # Using small GPT-2 for example
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token for GPT-2
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    # Prepare dataset
    tokenized_dataset = prepare_data(tokenizer)
    
    # Create dataloader
    train_dataloader = DataLoader(
        tokenized_dataset['train'],
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Configure BlockLLM optimizer
    config = BlockLLMConfig(
        lr=5e-5,  # Lower learning rate
        sparsity_level=0.5,  # Less sparsity - allow 50% of parameters to be active
        update_freq=1000,  # Less frequent parameter adjustments
        num_bottom_to_sample=5,  # Sample more parameters at once
        patience=100,  # More patience before adjusting
        param_update_interval=50  # Less frequent parameter swapping
    )
    # Initialize BlockLLM optimizer
    optimizer = BlockLLM(
        model.named_parameters(),
        config=config
    )

    # Lets just use AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Create learning rate scheduler
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=1000,  # More warmup steps
        num_training_steps=len(train_dataloader) * num_epochs
    )
    
    # Train the model
    train(model, train_dataloader, optimizer, num_epochs=num_epochs, device=device, scheduler=scheduler)

if __name__ == "__main__":
    main()