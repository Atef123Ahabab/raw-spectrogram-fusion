# Cell 2: Dataset Class for Feature Fusion

class SignalSpectrogramDataset(Dataset):
    """
    Dataset that returns both raw signal and spectrogram for each sample.
    
    Args:
        raw_signals: numpy array of shape (N, T) - raw time-domain signals
        spectrograms: numpy array of shape (N, 1, F, T_spec) - spectrograms
        labels: numpy array of shape (N,) - class labels
    """
    def __init__(self, raw_signals, spectrograms, labels):
        self.raw_signals = torch.FloatTensor(raw_signals)
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
        
        # Ensure spectrograms have channel dimension
        if self.spectrograms.ndim == 3:
            self.spectrograms = self.spectrograms.unsqueeze(1)  # Add channel dim
        
        assert len(self.raw_signals) == len(self.spectrograms) == len(self.labels), \
            "Raw signals, spectrograms, and labels must have the same length"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Returns:
            raw_signal: Tensor of shape (T,)
            spectrogram: Tensor of shape (1, F, T_spec)
            label: integer class label
        """
        return self.raw_signals[idx], self.spectrograms[idx], self.labels[idx]


# Example function to create spectrograms from raw signals
def create_spectrogram(signal, nperseg=64, noverlap=32):
    """
    Convert raw signal to spectrogram using STFT.
    
    Args:
        signal: 1D numpy array
        nperseg: length of each segment
        noverlap: number of points to overlap between segments
    
    Returns:
        spectrogram: 2D numpy array (F, T_spec)
    """
    from scipy import signal as scipy_signal
    
    f, t, Sxx = scipy_signal.spectrogram(signal, nperseg=nperseg, noverlap=noverlap)
    
    # Convert to log scale
    Sxx_log = np.log10(Sxx + 1e-10)
    
    return Sxx_log


# Helper function to prepare your data
# MODIFY THIS FUNCTION based on your Assignment 2 data loading code
def prepare_data_from_assignment2():
    """
    Replace this function with your actual data loading code from Assignment 2.
    
    This is a placeholder that shows the expected format.
    You should load your actual dataset here.
    
    Returns:
        train_raw, train_spec, train_labels
        val_raw, val_spec, val_labels
        test_raw, test_spec, test_labels
        num_classes
    """
    
    # ============================================
    # PLACEHOLDER: Replace with your actual data
    # ============================================
    
    # Example: synthetic data (REPLACE THIS)
    num_samples = 1000
    signal_length = 1000
    num_classes = 5
    
    # Generate random signals (replace with your data)
    all_signals = np.random.randn(num_samples, signal_length).astype(np.float32)
    all_labels = np.random.randint(0, num_classes, num_samples)
    
    # Create spectrograms from signals
    spectrograms = []
    for signal in all_signals:
        spec = create_spectrogram(signal)
        spectrograms.append(spec)
    all_spectrograms = np.array(spectrograms)
    
    # Train/val/test split (70/15/15)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    
    indices = np.random.permutation(num_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_raw = all_signals[train_idx]
    train_spec = all_spectrograms[train_idx]
    train_labels = all_labels[train_idx]
    
    val_raw = all_signals[val_idx]
    val_spec = all_spectrograms[val_idx]
    val_labels = all_labels[val_idx]
    
    test_raw = all_signals[test_idx]
    test_spec = all_spectrograms[test_idx]
    test_labels = all_labels[test_idx]
    
    print(f"Dataset shapes:")
    print(f"Train - Raw: {train_raw.shape}, Spec: {train_spec.shape}, Labels: {train_labels.shape}")
    print(f"Val   - Raw: {val_raw.shape}, Spec: {val_spec.shape}, Labels: {val_labels.shape}")
    print(f"Test  - Raw: {test_raw.shape}, Spec: {test_spec.shape}, Labels: {test_labels.shape}")
    print(f"Number of classes: {num_classes}")
    
    return (train_raw, train_spec, train_labels,
            val_raw, val_spec, val_labels,
            test_raw, test_spec, test_labels,
            num_classes)
