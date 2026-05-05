# Cell 4: Training and Evaluation Functions

def train_epoch(model, dataloader, criterion, optimizer, device, model_type='fusion'):
    """
    Train for one epoch.
    
    Args:
        model: the neural network model
        dataloader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: cuda or cpu
        model_type: 'raw', 'spec', or 'fusion'
    
    Returns:
        average loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for raw_signal, spectrogram, labels in tqdm(dataloader, desc='Training', leave=False):
        raw_signal = raw_signal.to(device)
        spectrogram = spectrogram.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass based on model type
        if model_type == 'raw':
            outputs = model(raw_signal)
        elif model_type == 'spec':
            outputs = model(spectrogram=spectrogram)
        else:  # fusion
            outputs = model(raw_signal, spectrogram)
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, model_type='fusion'):
    """
    Evaluate the model.
    
    Args:
        model: the neural network model
        dataloader: validation/test data loader
        criterion: loss function
        device: cuda or cpu
        model_type: 'raw', 'spec', or 'fusion'
    
    Returns:
        average loss, accuracy, all predictions, all labels
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for raw_signal, spectrogram, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            raw_signal = raw_signal.to(device)
            spectrogram = spectrogram.to(device)
            labels = labels.to(device)
            
            # Forward pass based on model type
            if model_type == 'raw':
                outputs = model(raw_signal)
            elif model_type == 'spec':
                outputs = model(spectrogram=spectrogram)
            else:  # fusion
                outputs = model(raw_signal, spectrogram)
            
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_acc = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs, device, model_type='fusion', model_name='Model'):
    """
    Complete training loop with validation.
    
    Returns:
        trained model, history dictionary
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, model_type)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, 
                                          device, model_type)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%\n")
    
    return model, history
