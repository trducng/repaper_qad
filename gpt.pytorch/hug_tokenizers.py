"""Use HuggingFace tokenizers as it already has extra needed utilities"""
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def get_wkitext_tokenizer(path: str):
    """Get the tokenizer

    Args:
        path: the path that store tokenizer json file. If available, load. Otherwise
            start training the tokenizer from wkiki text

    Returns:
        The tokenizer
    """
    if Path(path).exists():
        return Tokenizer.from_file(path)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    files = [f"downloads/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
    tokenizer.train(files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    tokenizer.save(path)

    return tokenizer


def get_bookcorpus_tokenizer(path: str):
    """Get the tokenizer

    Args:
        path: the path that store tokenizer json file. If available, load. Otherwise
            start training the tokenizer from wkiki text

    Returns:
        The tokenizer
    """
    if Path(path).exists():
        return Tokenizer.from_file(path)

    def book_iterator():
        dataset = load_dataset("bookcorpus")
        for item in dataset['train']:
            yield item['text']

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(book_iterator(), trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    tokenizer.save(path)

    return tokenizer


if __name__ == '__main__':
    texts = [
        "Hello darkness my old friend!!!! My name's Harry Potter".lower(),
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles".lower()
    ]
    tokenizer = get_bookcorpus_tokenizer("downloads/tokenizer_bookcorpus.json")
    output = tokenizer.encode_batch(texts)
    print(f"{texts[0]=}")
    print(f"{output[0].ids=}")
    print(f"{output[0].tokens=}")
    print(f"{output[0].type_ids=}")
    print(f"{texts[1]=}")
    print(f"{output[1].ids=}")
    print(f"{output[1].tokens=}")
    print(f"{output[1].type_ids=}")

    # print()
    # from transformers import OpenAIGPTTokenizer
    # tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    # output = tokenizer.encode(text)
    # print(f"{text=}")
    # import pdb; pdb.set_trace()
    # print(f"{output=}")
