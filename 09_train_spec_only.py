# Cell 9: Train Spectrogram-Only Model

print("\n" + "="*70)
print("TRAINING MODEL 2: SpecOnlyModel")
print("="*70)

# Initialize model
spec_only_model = SpecOnlyModel(num_classes, emb_dim).to(device)

# Loss and optimizer
optimizer_spec = torch.optim.Adam(spec_only_model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=weight_decay)
scheduler_spec = torch.optim.lr_scheduler.StepLR(optimizer_spec, step_size=10, gamma=0.5)

# Train
spec_only_model, spec_history = train_model(
    spec_only_model, train_loader, val_loader, 
    criterion, optimizer_spec, scheduler_spec, 
    num_epochs, device, model_type='spec', model_name='SpecOnlyModel'
)

print("✓ SpecOnlyModel training complete!\n")
