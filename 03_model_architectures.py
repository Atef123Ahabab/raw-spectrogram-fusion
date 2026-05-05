# Cell 3: Model Architectures

class RawBranch1D(nn.Module):
    """
    1D CNN for processing raw time-domain signals.
    Input shape: (B, T)
    Output shape: (B, emb_dim)
    """
    def __init__(self, signal_length, emb_dim=128):
        super(RawBranch1D, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer to embedding
        self.fc = nn.Linear(256, emb_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Args:
            x: (B, T) raw signals
        Returns:
            embedding: (B, emb_dim)
        """
        # Add channel dimension: (B, T) -> (B, 1, T)
        x = x.unsqueeze(1)
        
        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling: (B, 256, T') -> (B, 256, 1) -> (B, 256)
        x = self.global_pool(x).squeeze(-1)
        
        # Embedding
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class SpecBranch2D(nn.Module):
    """
    2D CNN for processing spectrograms.
    Input shape: (B, 1, F, T_spec)
    Output shape: (B, emb_dim)
    """
    def __init__(self, emb_dim=128):
        super(SpecBranch2D, self).__init__()
        
        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer to embedding
        self.fc = nn.Linear(256, emb_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, F, T_spec) spectrograms
        Returns:
            embedding: (B, emb_dim)
        """
        # Convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling: (B, 256, F', T') -> (B, 256, 1, 1) -> (B, 256)
        x = self.global_pool(x).view(x.size(0), -1)
        
        # Embedding
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class FusionHead(nn.Module):
    """
    Fusion head that concatenates embeddings and classifies.
    Input: two embeddings of shape (B, emb_dim) each
    Output: logits of shape (B, num_classes)
    """
    def __init__(self, emb_dim, num_classes):
        super(FusionHead, self).__init__()
        
        # Input will be concatenated: 2 * emb_dim
        self.fc1 = nn.Linear(2 * emb_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, z_raw, z_spec):
        """
        Args:
            z_raw: (B, emb_dim) embeddings from raw branch
            z_spec: (B, emb_dim) embeddings from spec branch
        Returns:
            logits: (B, num_classes)
        """
        # Concatenate embeddings
        z = torch.cat([z_raw, z_spec], dim=1)  # (B, 2*emb_dim)
        
        # Classification layers
        z = F.relu(self.bn1(self.fc1(z)))
        z = self.dropout1(z)
        
        z = F.relu(self.bn2(self.fc2(z)))
        z = self.dropout2(z)
        
        logits = self.fc3(z)
        
        return logits


class RawOnlyModel(nn.Module):
    """Complete model using only raw signals."""
    def __init__(self, signal_length, num_classes, emb_dim=128):
        super(RawOnlyModel, self).__init__()
        self.raw_branch = RawBranch1D(signal_length, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        
    def forward(self, raw_signal, spectrogram=None):
        """
        Args:
            raw_signal: (B, T)
            spectrogram: ignored (for compatibility)
        Returns:
            logits: (B, num_classes)
        """
        z_raw = self.raw_branch(raw_signal)
        logits = self.classifier(z_raw)
        return logits


class SpecOnlyModel(nn.Module):
    """Complete model using only spectrograms."""
    def __init__(self, num_classes, emb_dim=128):
        super(SpecOnlyModel, self).__init__()
        self.spec_branch = SpecBranch2D(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        
    def forward(self, raw_signal=None, spectrogram=None):
        """
        Args:
            raw_signal: ignored (for compatibility)
            spectrogram: (B, 1, F, T_spec)
        Returns:
            logits: (B, num_classes)
        """
        z_spec = self.spec_branch(spectrogram)
        logits = self.classifier(z_spec)
        return logits


class TwoStreamFusionModel(nn.Module):
    """Complete two-stream fusion model."""
    def __init__(self, signal_length, num_classes, emb_dim=128):
        super(TwoStreamFusionModel, self).__init__()
        self.raw_branch = RawBranch1D(signal_length, emb_dim)
        self.spec_branch = SpecBranch2D(emb_dim)
        self.fusion_head = FusionHead(emb_dim, num_classes)
        
    def forward(self, raw_signal, spectrogram):
        """
        Args:
            raw_signal: (B, T)
            spectrogram: (B, 1, F, T_spec)
        Returns:
            logits: (B, num_classes)
        """
        z_raw = self.raw_branch(raw_signal)
        z_spec = self.spec_branch(spectrogram)
        logits = self.fusion_head(z_raw, z_spec)
        return logits
