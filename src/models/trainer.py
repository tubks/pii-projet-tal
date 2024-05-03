import torch 

from transformers import Trainer


class PIITrainer(Trainer):
    
    def __init__(self, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        labels = inputs["labels"]
        outputs = model(**inputs)   
        criterion = torch.nn.CrossEntropyLoss(weight=self.weights)
        loss = criterion(outputs['logits'].permute(0, 2, 1), labels)
        return (loss, outputs) if return_outputs else loss
