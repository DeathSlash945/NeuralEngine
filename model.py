import torch
import torch.nn as nn

class NeuralPersonalityRVAE(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, latent_dim=16, trait_dim=5, action_dim=3):
        super(NeuralPersonalityRVAE, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        #projecting inputs
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU()
        )
        
        # the lstm
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_dim, batch_first=True)
        
        # Variational Bottleneck Heads (Mapping LSTM output to Distribution Parameters)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        #personality traits of the model
        self.personality_decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, trait_dim),
            nn.Sigmoid()
        )
        
        #behaviour action selection is done
        self.action_decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )

    def reparameterize(self, mu, logvar):
        #reparam trick of var-autoencoders, shifting randomness to external param to control grads
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        #x shape: [Batch, Seq_Len, Input_Dim]
        batch_size, seq_len, _ = x.size()
        
        embedded = self.input_embed(x)
        lstm_out, _ = self.lstm(embedded) # Shape: [Batch, Seq_Len, hidden_dim]
        
        # Squeeze the temporal outputs into distribution parameters at every time step
        mu = self.fc_mu(lstm_out) # Shape: [Batch, Seq_Len, latent_dim]
        logvar = self.fc_logvar(lstm_out)
        z = self.reparameterize(mu, logvar)
        
        # Decode the sampled latent states into our target spaces
        predicted_traits = self.personality_decoder(z)
        action_logits = self.action_decoder(z)
        
        return predicted_traits, action_logits, mu, logvar, z