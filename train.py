import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from transformer import Transformer
from Audioencoder import AudioEncoder
from data import loader
from torch.amp.grad_scaler import GradScaler
from config import config
from tqdm.auto import tqdm

grad_scaler = GradScaler(device='cuda')
device = 'cuda' if torch.cuda.is_available() else "cpu"

#hyperparameters
epochs = config["training"]["epochs"]
ar_lr = config["training"]["ar_lr"]
enc_lr = config["training"]["enc_lr"]


ar_model = Transformer(config["transformer"]).to(device)
audio_encoder = AudioEncoder(**config["audio_encoder"]).to(device)

#loss functions and optimizers
mse_loss = nn.MSELoss()
cross_loss = nn.CrossEntropyLoss()
optimizer_ar = optim.AdamW(ar_model.parameters(), lr =ar_lr)
optimizer_enc = optim.AdamW(audio_encoder.parameters(), lr =enc_lr)

# run = wandb.init(
#     entity="first_project",
#     project="seed_music",
#     config={ #need to be chnaged 
#         # "learning_rate": 0.02,
#         # "architecture": "CNN",
#         # "dataset": "CIFAR-100", 
#         # "epochs": 10,
#     },
# )

global_steps = 0
for epoch in range(epochs):
    ar_model.train()
    audio_encoder.train()
    for batch_idx, batch in enumerate(tqdm(loader,desc=f"Epoch {epoch}/{epochs}")):
        global_steps += 1
        
        audio = batch['audio'].to(device)
        text_ids = batch['input_ids'].to(device)
        
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            
            decoded, loss_cd, loss_enc, discrete_enc = audio_encoder(audio)
            discrete_enc = discrete_enc.to(device)
            reconst_loss = mse_loss(audio,decoded)
            loss_encoder = loss_cd + loss_enc + reconst_loss #encoder_loss
            
            t_s = text_ids.shape[1]
            logits = ar_model(discrete_enc,text_ids)
            audio_logits = logits[:,t_s:-1]
            targets = discrete_enc[:,1:]
            loss_ar = cross_loss(audio_logits.reshape(-1,logits.shape[-1]),targets.reshape(-1)) #ar_loss
            
            # run.log({"Encoder_loss": {"codebook": loss_cd, "enc": loss_enc, "reconst_loss": reconst_loss},
            #          "Transformer_loss": {"cross_en_loss": loss_ar}}, 
            #          steps = global_steps)
            
        optimizer_ar.zero_grad()
        optimizer_enc.zero_grad()
        
        grad_scaler.scale(loss_ar).backward()
        grad_scaler.scale(loss_encoder).backward()
        
        total_norm = 0
        for p in ar_model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        # run.log({"grad_norm": total_norm}, step=global_steps)

        grad_scaler.step(optimizer_ar)
        grad_scaler.step(optimizer_enc)
        
        grad_scaler.update()
        
        if batch_idx % 10 == 0:
            torch.save(ar_model.state_dict(),f"checkpoints/ar_model{global_steps}.pth")
            torch.save(audio_encoder.state_dict(),f"checkpoints/enc_model{global_steps}.pth")
            
        if global_steps % 1000 == 0:
            wandb.log({"audio_sample": wandb.Audio(decoded[0].cpu().numpy(), sample_rate=16000)})
