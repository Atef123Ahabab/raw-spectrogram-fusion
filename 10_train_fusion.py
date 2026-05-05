# Cell 10: Train Two-Stream Fusion Model

print("\n" + "="*70)
print("TRAINING MODEL 3: TwoStreamFusionModel")
print("="*70)

# Initialize model
fusion_model = TwoStreamFusionModel(signal_length, num_classes, emb_dim).to(device)

# Loss and optimizer
optimizer_fusion = torch.optim.Adam(fusion_model.parameters(), 
                                    lr=learning_rate, 
                                    weight_decay=weight_decay)
scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer_fusion, step_size=10, gamma=0.5)

# Train
fusion_model, fusion_history = train_model(
    fusion_model, train_loader, val_loader, 
    criterion, optimizer_fusion, scheduler_fusion, 
    num_epochs, device, model_type='fusion', model_name='TwoStreamFusionModel'
)

print("✓ TwoStreamFusionModel training complete!\n")
