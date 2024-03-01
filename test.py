class Encoder(nn.Module):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None
        # compute the flattened size after convolutions
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        # define fully connected layer to create embeddings
        self.fc = nn.Linear(flattened_size, embedding_dim)
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # store the shape before flattening
        self.shape_before_flattening = x.shape[1:]
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        x = self.fc(x)
        return x
    
class ConvAutoencoder1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvAutoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3)
        )
        self.fc = nn.Linear(128*2, output_dim)  # Adjust the dimension based on your input size
    
    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ConvAutoencoder1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvAutoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Linear(128*2, output_dim)  # Adjust the dimension based on your input size
    
    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x