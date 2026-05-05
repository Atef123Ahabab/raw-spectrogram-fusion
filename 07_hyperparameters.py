# Cell 7: Hyperparameters

# Training hyperparameters
emb_dim = 128  # Embedding dimension for both branches
num_epochs = 30
learning_rate = 0.001
weight_decay = 1e-4

print(f"\nTraining Configuration:")
print(f"Embedding dimension: {emb_dim}")
print(f"Number of epochs: {num_epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Weight decay: {weight_decay}")
print(f"Device: {device}")
