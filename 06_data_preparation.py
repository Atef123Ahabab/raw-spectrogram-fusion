# Cell 6: Data Preparation
# Load and prepare your data

# IMPORTANT: Replace the prepare_data_from_assignment2() function 
# with your actual data loading code from Assignment 2

(train_raw, train_spec, train_labels,
 val_raw, val_spec, val_labels,
 test_raw, test_spec, test_labels,
 num_classes) = prepare_data_from_assignment2()

# Create datasets
train_dataset = SignalSpectrogramDataset(train_raw, train_spec, train_labels)
val_dataset = SignalSpectrogramDataset(val_raw, val_spec, val_labels)
test_dataset = SignalSpectrogramDataset(test_raw, test_spec, test_labels)

# Create data loaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Get data dimensions
signal_length = train_raw.shape[1]

print(f"\nDataset Information:")
print(f"Signal length: {signal_length}")
print(f"Number of classes: {num_classes}")
print(f"Batch size: {batch_size}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Create class names (customize based on your dataset)
class_names = [f'Class {i}' for i in range(num_classes)]
print(f"Class names: {class_names}")
