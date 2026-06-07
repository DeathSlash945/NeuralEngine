import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from model import NeuralPersonalityRVAE

def train_rvae():
    BATCH_SIZE = 64
    EPOCHS = 25
    LEARNING_RATE = 0.003
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.load("data/X_events.pt")
    Y_p = torch.load("data/Y_personality.pt")
    Y_a = torch.load("data/Y_actions.pt")

    dataset = TensorDataset(X, Y_p, Y_a)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize our variational model
    model = NeuralPersonalityRVAE(latent_dim=16).to(DEVICE)
    
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training variational latent (R-VAE) closed-Loop model...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        
        for batch_X, batch_Y_p, batch_Y_a in train_loader:
            batch_X, batch_Y_p, batch_Y_a = batch_X.to(DEVICE), batch_Y_p.to(DEVICE), batch_Y_a.to(DEVICE)
            
            pred_p, pred_logits, mu, logvar, _ = model(batch_X)
            
            # std reconstruction loss computed
            loss_p = mse_criterion(pred_p, batch_Y_p)
            loss_a = ce_criterion(pred_logits.view(-1, 3), batch_Y_a.view(-1))
            reconstruction_loss = loss_p + loss_a
            
            # KL divergence loss for normal dist
            # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) for ref 
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = reconstruction_loss + 0.01 * kl_loss    #scaling with the loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item() * batch_X.size(0)
            epoch_kl += kl_loss.item() * batch_X.size(0)
        epoch_loss /= len(train_loader.dataset)
        epoch_kl /= len(train_loader.dataset)
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Total Loss: {epoch_loss:.5f} | Raw KL Loss: {epoch_kl:.4f}")

    torch.save(model.state_dict(), "data/personality_engine.pth")
    print("Done!")

if __name__ == "__main__":
    train_rvae()