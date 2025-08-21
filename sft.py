import os
import math
import random
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset, DataLoader

def get_lr(it, max_steps, warmup_steps, max_lr, min_lr):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

class LanguageModel:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(device)

    def configure_optimizer(self, weight_decay, max_lr):
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=max_lr)
        return optimizer

    def generate_samples(self, dataset, sample_indices, num_samples=16):
        print(f"\nGenerating {num_samples} samples for evaluation...\n")
        self.model.eval()

        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            question = sample["question"]
            ground_truth_answer = sample["answer"]

            chat_prompt = [{'role': 'user', 'content': question}]
            prompt_text = self.tokenizer.apply_chat_template(
                chat_prompt,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            prompt_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0][prompt_length:]
            model_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            print(f"--- SAMPLE {i+1}/{num_samples} (Example #{idx}) ---")
            print(f"QUESTION:\n{question}\n")
            print(f"GROUND TRUTH:\n{ground_truth_answer}\n")
            print(f"MODEL OUTPUT:\n{model_answer.strip()}\n")
            print("="*50 + "\n")

        self.model.train()
        print("Generation complete! Resuming training...")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

max_grad_norm = 1.0
batch_size = 2
gradient_accumulation_steps = 8
num_epochs = 4
weight_decay = 0.1
max_lr = 3e-5
min_lr = 3e-6
warmup_ratio = 0.1

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
lm = LanguageModel(model_name, device)
lm.model.train()
tokenizer = lm.tokenizer

optimizer = lm.configure_optimizer(weight_decay, max_lr)
optimizer.zero_grad(set_to_none=True)

log_dir = "./sft_model_logs_improved"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f: pass

checkpoint_dir = os.path.join(log_dir, "latest_checkpoint")
os.makedirs(checkpoint_dir, exist_ok=True)


train_data = load_dataset("gsm8k", "main", split="train")

class SFTDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["question"], item["answer"]

def collate_fn(batch):
    input_ids_list, labels_list = [], []

    for question, answer in batch:
        chat_prompt = [
            {'role': 'user', 'content': question}
        ]
        prompt_str = tokenizer.apply_chat_template(
            chat_prompt, tokenize=False, add_generation_prompt=True
        )

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = tokenizer.encode(prompt_str + answer + tokenizer.eos_token, add_special_tokens=False)

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))

        labels = ([-100] * len(prompt_ids)) + full_ids[len(prompt_ids):]
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return input_ids, attention_mask, labels

train_dataset = SFTDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

num_eval_samples = 16
total_examples = len(train_data)
eval_sample_indices = random.sample(range(total_examples), num_eval_samples)


total_steps = len(train_loader) // gradient_accumulation_steps * num_epochs
warmup_steps = int(warmup_ratio * total_steps)
print(f"Starting training for {num_epochs} epoch(s). Total steps: {total_steps}")

global_step = 0
epsilon = 1e-8

for epoch in range(num_epochs):
    lm.model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for step, (input_ids, attention_mask, labels) in enumerate(pbar):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = lm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(lm.model.parameters(), max_grad_norm)
            lr = get_lr(global_step, total_steps, warmup_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item() * gradient_accumulation_steps
            pbar.set_postfix({"custom_loss": f"{loss_val:.4f}", "lr": f"{lr:.2e}"})
            with open(log_file, "a") as f:
                f.write(f'Epoch {epoch+1}, Step {global_step}, Loss: {loss_val:.4f}, LR: {lr:.2e}\n')

            global_step += 1

    print(f"\nSaving checkpoint for epoch {epoch+1} to {checkpoint_dir}...")
    lm.model.save_pretrained(checkpoint_dir)
    lm.tokenizer.save_pretrained(checkpoint_dir)
    print(f"Checkpoint saved successfully.")

    lm.generate_samples(
        dataset=train_data,
        sample_indices=eval_sample_indices,
        num_samples=num_eval_samples
    )


print("Training finished!")
