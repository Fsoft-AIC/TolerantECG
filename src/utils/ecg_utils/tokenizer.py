from transformers import AutoTokenizer


class HuggingFaceTokenizer:
    def __init__(self, model_name='michiyasunaga/BioLinkBERT-base') -> None:
        self.model_name = model_name
        if model_name == "emilyalsentzer/Bio_ClinicalBERT":
            self.max_length = 128
        else:
            self.max_length = 512

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts):
        # texts: a list of sentences
        return self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

    def batch_decode(self, token_ids):
        """
        token_ids: [Batch_size, Seq_len] 
        return: a list of string senteces
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)

    def decode(self, token_ids):
        """
        token_ids: 1D list representing ids of tokens
        return: a string sentence
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
