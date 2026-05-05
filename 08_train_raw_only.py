# Cell 8: Train Raw-Only Model

print("\n" + "="*70)
print("TRAINING MODEL 1: RawOnlyModel")
print("="*70)

# Initialize model
raw_only_model = RawOnlyModel(signal_length, num_classes, emb_dim).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_raw = torch.optim.Adam(raw_only_model.parameters(), 
                                 lr=learning_rate, 
                                 weight_decay=weight_decay)
scheduler_raw = torch.optim.lr_scheduler.StepLR(optimizer_raw, step_size=10, gamma=0.5)

# Train
raw_only_model, raw_history = train_model(
    raw_only_model, train_loader, val_loader, 
    criterion, optimizer_raw, scheduler_raw, 
    num_epochs, device, model_type='raw', model_name='RawOnlyModel'
)

print("✓ RawOnlyModel training complete!\n")
