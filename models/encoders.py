import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer, T5Tokenizer, T5EncoderModel


class ProteinEncoder(nn.Module):
    def __init__(self, esm_model="/root/autodl-tmp/models/esm2_t33_650M_UR50D",
                 d_model=768, device="cuda",
                 backbone_obj=None, train_backbone=False,
                 tokenizer_obj=None):
        super().__init__()
        self.device = device
        try:
            from transformers import EsmTokenizer
            self.tokenizer = tokenizer_obj or EsmTokenizer.from_pretrained(esm_model, local_files_only=False)
        except Exception:
            from transformers import AutoTokenizer
            self.tokenizer = tokenizer_obj or AutoTokenizer.from_pretrained(esm_model, local_files_only=False)

        if backbone_obj is not None:
            self.backbone = backbone_obj.to(device)
        else:
            try:
                from transformers import EsmModel
                self.backbone = EsmModel.from_pretrained(esm_model, local_files_only=False).to(device)
            except Exception:
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained(esm_model, local_files_only=False).to(device)

        self.train_backbone = train_backbone
        if not self.train_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, d_model).to(device)

    def encode_tokens(self, seqs):
        inputs = self.tokenizer(seqs, return_tensors="pt", padding=True,
                                truncation=True, max_length=640) # 768 
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.set_grad_enabled(self.train_backbone):
            outputs = self.backbone(**inputs).last_hidden_state  # (B,L,H)

        mask = inputs["attention_mask"].bool()
        return outputs, mask

    def forward(self, seqs):
        tok, mask = self.encode_tokens(seqs)
        tok = self.proj(tok)
        return tok, mask
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel


class LigandEncoder(nn.Module):
    def __init__(self, molt5_model="/root/autodl-tmp/models/molt5-base-smiles2caption",
                 d_model=768, device="cuda",
                 backbone_obj=None, train_backbone=False,
                 tokenizer_obj=None):
        super().__init__()
        self.device = device
        from transformers import T5Tokenizer, T5EncoderModel

        self.tokenizer = tokenizer_obj or T5Tokenizer.from_pretrained(molt5_model, local_files_only=False)

        if backbone_obj is not None:
            self.backbone = backbone_obj.to(device)  # 例如 self.molt5.get_encoder()
        else:
            self.backbone = T5EncoderModel.from_pretrained(molt5_model, local_files_only=False).to(device)

        self.train_backbone = train_backbone
        if not self.train_backbone:
            for name, p in self.backbone.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)

            has_trainable_lora = any(
                ("lora_" in n) and p.requires_grad for n, p in self.backbone.named_parameters()
            )
            if has_trainable_lora:
                self.backbone.train()
            else:
                self.backbone.eval()
        else:
            self.backbone.train()

        hidden = self.backbone.config.d_model
        self.proj = nn.Linear(hidden, d_model).to(device)

    def encode_tokens(self, smiles):
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True,
                                truncation=True, max_length=384) # 384
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        enable_grad = any(p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(enable_grad):
            outputs = self.backbone(**inputs).last_hidden_state
            
        outputs = torch.nan_to_num(outputs)
        mask = inputs["attention_mask"].bool()
        return outputs, mask

    def forward(self, smiles):
        tok, mask = self.encode_tokens(smiles)
        tok = self.proj(tok.float())
        tok = torch.nan_to_num(tok)
        return tok, mask
