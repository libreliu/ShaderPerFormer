from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from dataset.FragmentPerformanceDataset import FragmentPerformanceDataset

def train(outputFile):
    # BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Decoder
    tokenizer.decoder = decoders.BPEDecoder()

    # Trainer
    trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2, special_tokens=[
        "[BOS]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"
    ])

    def corpus_iterator():
        dset = FragmentPerformanceDataset(None)
        for i in range(0, len(dset)):
            yield dset[i]["spvText"]

    # Train
    tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)

    # Save
    tokenizer.save(outputFile)