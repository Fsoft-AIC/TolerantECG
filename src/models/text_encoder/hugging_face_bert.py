import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class HuggingFaceBert(nn.Module):
    def __init__(self, 
                 model_name='michiyasunaga/BioLinkBERT-base',
                 num_classes: int = 768, 
                 is_flash_attn: bool = True,
                 is_finetune: bool = True, 
                 head_bias=False,
                 last_norm=False,
                 ) -> None:
        super().__init__()

        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_config(config)

        if is_flash_attn:
            from src.models.text_encoder.bert_flash_attention import BertFlashAttnetion
            for layer in self.model.encoder.layer:
                layer.attention.self = BertFlashAttnetion(config)

        pretrained_model = AutoModel.from_pretrained(model_name)
        self.model.load_state_dict(pretrained_model.state_dict())

        self.model.pooler = nn.Identity()

        if is_finetune:
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.norm = nn.LayerNorm(768) if last_norm else nn.Identity()
        self.head = nn.Linear(768, num_classes, bias=head_bias) if num_classes > 0 else nn.Identity()

        self.embed_dim = 768
        self.out_dim = num_classes if num_classes > 0 else self.embed_dim
        self.head_bias = head_bias
        self.last_norm = last_norm

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.model(input_ids=input_ids, 
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        return output.last_hidden_state, self.head(self.norm(output.last_hidden_state[:, 0, :]))
    