import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("m-aliabbas/idrak_timit_subsample1",split="train")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class TrainDataset(Dataset):
    def __init__(self, data, text_tokenizer, max_audio_samples=25000, max_text_length=100):
        super().__init__()
        
        self.data = data
        self.tokenizer = text_tokenizer
        self.max_audio_samples = max_audio_samples
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["transcription"]
        audio = item["audio"]['array']

        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        if audio_tensor.shape[0] > self.max_audio_samples:
            audio_tensor = audio_tensor[:self.max_audio_samples]
        elif audio_tensor.shape[0] < self.max_audio_samples:
            padding = self.max_audio_samples - audio_tensor.shape[0]
            audio_tensor = torch.nn.pad(audio_tensor, (0, padding))

        return {
            "input_ids": input_ids,
            "audio": audio_tensor.unsqueeze(0)
        }


dataset = TrainDataset(data = ds, text_tokenizer=tokenizer, max_audio_samples=25000, max_text_length=100)
loader = DataLoader(dataset=dataset, batch_size=32, pin_memory=True, shuffle=True)