import json
import pickle
from collections import defaultdict

import regex as re
from tqdm import tqdm


PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_pairs_frequency(vocab):
    """Get the pair frequency

    Args:
        vocab: {
            ((x, y), (z,), (a, b)): 10
        }
    """
    freq = defaultdict(int)
    cache = defaultdict(list)
    for word_idx, each_word in enumerate(vocab.keys()):
        for idx in range(len(each_word) - 1):
            freq[each_word[idx], each_word[idx + 1]] += vocab[each_word]
            cache[each_word[idx], each_word[idx + 1]].append(word_idx)
    return freq, cache


def visualize(byte):
    """Visualize characters with byte representation"""
    return "".join([chr(each) for each in byte])


def load_mingpt_bpe(encoder, vocab_merge):
    """Load GPT BPE

    Args:
        encoder: encoder file
        vocab_merge: the vocab merge file
    """
    def bytes_to_unicode() -> dict:
        """Taken from karpathy's minGPT
        Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
        character that represents it visually. Some bytes have their appearance preserved
        because they don't cause any trouble. These are defined in list bs. For example:
        chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
        However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
        bytes, into new characters in a range where chr() returns a single nice character.
        So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
        In particular, the space character is 32, which we can see by ord(' '). Instead,
        this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
        So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
        that "look nice", either in their original form, or a funny shifted character
        like 'Ā', or 'Ġ', etc.
        """
        # the 188 integers that render fine in their original form and need no shifting
        bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:] # all integers b in bs will simply map to chr(b) in the output dict
        # now get the representations of the other 68 integers that do need shifting
        # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
        n = 0
        for b in range(2**8):
            if b not in bs:
                # if this byte is "ugly" then map it to the next available "nice" character
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        d = dict(zip(bs, cs))
        return d

    bytes_decoder = {ord(value): key for key, value in bytes_to_unicode().items()}

    with open(encoder, "r") as fi:
        vocab = {}
        for token, idx in json.load(fi).items():
            encoded_token = [bytes_decoder[ord(each)] for each in token]
            vocab[tuple(encoded_token)] = idx
        vocab_reverse = {value: key for key, value in vocab.items()}

    with open(vocab_merge, "r") as fi:
        content = fi.read().splitlines()
        vocab_rank = {}
        for idx, line in enumerate(content):
            first, second = line.split()
            encoded_first = [bytes_decoder[ord(each)] for each in first]
            encoded_second = [bytes_decoder[ord(each)] for each in second]
            vocab_rank[(tuple(encoded_first), tuple(encoded_second))] = idx

    bpe = BPE(vocab_size=len(vocab))
    bpe.vocab = vocab
    bpe.vocab_reverse = vocab_reverse
    bpe.vocab_rank = vocab_rank

    return bpe


class BPE:
    """Byte-pair encoding implementation
    ## Construct BPE files
    ## Load, encode, decode from BPE
    """

    def __init__(self, vocab_size: int):
        """Initialize a new BPE object

        Args:
            vocab_size: the size of the result vocabulary
        """
        self.vocab_size = vocab_size
        self.vocab = {}  # (b1, b2, b3): index
        self.vocab_reverse = {}  # index: (b1, b2, b3)
        self.vocab_frequency = {}  # ((b1, b2), (b3,)): freq
        self.vocab_rank = {}  # ((b1, b2), (b3,)): rank
        self.corpus = defaultdict(int)  # list of words

    @classmethod
    def load(cls, path):
        """Load the saved vocabulary"""
        with open(path, "rb") as fi:
            data = pickle.load(fi)

        bpe = cls(vocab_size=data["vocab_size"])
        bpe.vocab = data["vocab"]
        bpe.vocab_reverse = data["vocab_reverse"]
        bpe.vocab_frequency = data["vocab_frequency"]
        bpe.vocab_rank = data["vocab_rank"]
        bpe.corpus = data["corpus"]
        return bpe

    def save(self, path):
        """Save as a json file"""
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "vocab_reverse": self.vocab_reverse,
            "vocab_frequency": self.vocab_frequency,
            "vocab_rank": self.vocab_rank,
            "corpus": self.corpus,
        }
        with open(path, "wb") as fo:
            pickle.dump(data, fo)

    def encode(self, line: str, debug=False) -> list:
        """Encode a string of text into machine readable numbers

        Args:
            line: a string of text line

        Returns:
            list of items
        """
        tokens = re.findall(PAT, line)
        byte_idx = []
        for token in tokens:
            token_bytes = [(each,) for each in token.encode("utf-8")]
            pairs = [
                (token_bytes[idx], token_bytes[idx + 1])
                for idx in range(len(token_bytes) - 1)
            ]

            while True:
                if not pairs:
                    break
                matched = min(
                    pairs, key=lambda obj: self.vocab_rank.get(obj, float("inf"))
                )
                if matched not in self.vocab_rank:
                    break
                combined = matched[0] + matched[1]

                new_token_bytes = []
                i = 0
                while i < len(token_bytes):
                    if i >= len(token_bytes) - 1:
                        new_token_bytes.append(token_bytes[i])
                        i += 1
                        break
                    if token_bytes[i] != matched[0]:
                        new_token_bytes.append(token_bytes[i])
                        i += 1
                        continue
                    if token_bytes[i + 1] != matched[1]:
                        new_token_bytes.append(token_bytes[i])
                        i += 1
                        continue
                    new_token_bytes.append(combined)
                    i += 2

                token_bytes = new_token_bytes
                pairs = [
                    (token_bytes[idx], token_bytes[idx + 1])
                    for idx in range(len(token_bytes) - 1)
                ]

            # convert from token_bytes to byte_idx
            if debug:
                print(token, token_bytes, [self.vocab[each] for each in token_bytes])
            byte_idx.extend([self.vocab[each] for each in token_bytes])

        return byte_idx

    def decode(self, byte_idx: list) -> str:
        """Decode machine readable numbers into a string of text"""
        result = []
        for each_idx in byte_idx:
            token = self.vocab_reverse[each_idx]
            result.append(visualize(token))
        return "".join(result)

    def construct(self, corpus: list[str]):
        """Construct the vocab given a corpus

        Args:
            corpus: text
        """
        for line in corpus:
            tokens = re.findall(PAT, line)
            for token in tokens:
                token_bytes = tuple((each,) for each in token.encode("utf-8"))
                self.corpus[token_bytes] += 1
        for idx in range(256):
            self.vocab[(idx,)] = len(self.vocab)

        pbar = tqdm(total=self.vocab_size)
        pbar.update(len(self.vocab))
        while len(self.vocab) < self.vocab_size:
            pairs, cache = get_pairs_frequency(self.corpus)
            most_frequent = max(pairs, key=pairs.get)
            combined = most_frequent[0] + most_frequent[1]
            self.vocab[combined] = len(self.vocab)
            self.vocab_frequency[most_frequent] = pairs[most_frequent]
            self.vocab_rank[most_frequent] = len(self.vocab_rank)
            pbar.update(1)

            to_update = set(cache[most_frequent])
            change = {}
            for idx, each_word in enumerate(self.corpus.keys()):
                if idx not in to_update:
                    continue
                new = []
                i = 0
                while i < len(each_word):
                    if i >= len(each_word) - 1:
                        new.append(each_word[i])
                        break
                    if each_word[i] != most_frequent[0]:
                        new.append(each_word[i])
                        i += 1
                        continue
                    if each_word[i + 1] != most_frequent[1]:
                        new.append(each_word[i])
                        i += 1
                        continue
                    new.append(combined)
                    i += 2
                change[each_word] = tuple(new)
            for key, value in change.items():
                self.corpus[value] = self.corpus[key]
                del self.corpus[key]
        pbar.close()
        self.vocab_reverse = {value: key for key, value in self.vocab.items()}


if __name__ == "__main__":

    import sys
    import traceback
    from pdb import Pdb

    pdb = Pdb()

    try:
        ## Construction
        # bpe = BPE(vocab_size=2000)
        # with open("downloads/mind.txt", "r") as fi:
        #     lines = fi.read().splitlines()
        # bpe.construct(corpus=lines)
        # bpe.save("downloads/vocab_size_2000.json")

        ## Test
        # bpe = BPE.load("downloads/vocab_size_2000.json")
        bpe = load_mingpt_bpe(
            encoder="downloads/mingpt/encoder.json",
            vocab_merge="downloads/mingpt/vocab.bpe"
        )
        text = "Hello darkness my old friend!!! My name's Harry Potter"
        encoded = bpe.encode(text)
        decoded = bpe.decode(encoded)
        print("Text:", text)
        print("Encoded:", encoded)
        print("Decoded:", decoded)
        text = "Hellodarknessmyoldfriend!!!Myname'sHarryPotter"
        encoded = bpe.encode(text)
        decoded = bpe.decode(encoded)
        print("Text:", text)
        print("Encoded:", encoded)
        print("Decoded:", decoded)
    except Exception:
        traceback.print_exc()
        print("Uncaught exception. Entering post mortem debugging")
        t = sys.exc_info()[2]
        pdb.interaction(None, t)
