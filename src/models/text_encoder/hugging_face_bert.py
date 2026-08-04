import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class HuggingFaceBert(nn.Module):
    def __init__(self, 
                 model_name='michiyasunaga/BioLinkBERT-base',
                 is_finetune: bool = False, 
                 ) -> None:
        super().__init__()

        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_config(config)

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

        self.embed_dim = getattr(config, "hidden_size", getattr(config, "d_model", 768))

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.model(input_ids=input_ids, 
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        return output.last_hidden_state, output.last_hidden_state[:, 0, :]
    