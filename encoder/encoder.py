import torch
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

class Model(torch.nn.Module):
    def __init__(self, n_mels=80, embedding_dim=192, channel=512):
        super(Model, self).__init__()
        channels = [channel for _ in range(4)]
        channels.append(channel * 3)
        self.model = ECAPA_TDNN(input_size=n_mels, lin_neurons=embedding_dim, channels=channels)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.model(x)
        x = x.squeeze(1)
        return x

if __name__ == '__main__':
    # Fixing the naming issue for 'channel'
    model = Model(n_mels=80, embedding_dim=192, channel=1024)

    # Load the pretrained model checkpoint
    checkpoint = torch.load("./embedding_model.ckpt")
    
    new_state_dict = {f"model.{k}": v for k, v in checkpoint.items()}
    
    # Assuming the checkpoint contains the state dict directly
    model.load_state_dict(new_state_dict)

    # To evaluate or use the model
    model.eval()

    # Test with dummy input (B, 1, n_mels, T)
    dummy_input = torch.randn(1, 1, 300, 80)  # Example input
    output = model(dummy_input)
    print(output.shape)