import transformers
import torch.utils.data
from transformers import DataCollatorForLanguageModeling
from dataset.MapDataset import MapDataset

class SpvMaskedLanguageModelingTask:
    model: transformers.PreTrainedModel

    def __init__(self, model):
        self.model = model

    def setup_trainer(
        self,
        tokenizer: 'transformers.PreTrainedTokenizer',
        training_args: 'transformers.TrainingArguments',
        dataset: 'torch.utils.data.Dataset'
    ):

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        def dataset_postprocess_fn(elem):
            text = elem["spvText"]
            return text

        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=MapDataset(dataset, lambda elem: dataset_postprocess_fn(elem))
        )

        return trainer