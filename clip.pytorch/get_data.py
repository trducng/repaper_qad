import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from CLIP import CLIP, build_transform
from clip_tokenizer import SimpleTokenizer


def get_imagenet():
    # configs
    MODEL_PATH = 'clip.pth'
    VOCAB_PATH = 'bpe_simple_vocab_16e6.txt.gz'
    IMAGENET_PATH = '/home/john/datasets/imagenet/object_localization'
    is_fp16 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize the model
    model = CLIP(attention_probs_dropout_prob=0, hidden_dropout_prob=0)
    model.load_state_dict(state_dict = torch.load(MODEL_PATH))
    if is_fp16:
        model.to(device=device).eval().half()
    else:
        model.to(device=device).eval().float()

    # initializer the tokenizer + image transform
    tokenizer = SimpleTokenizer(
            bpe_path=VOCAB_PATH,
            context_length=model.context_length.item())
    transform = build_transform(model.input_resolution.item())

    # initialize the data
    data = datasets.ImageNet(IMAGENET_PATH, 'val', transform=transform)
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=16)
    # important no shuffle

    # inference
    predictions = []
    ground_truths = []
    with torch.no_grad():
        query = [f'a {", ".join(each)}' for each in data.classes]
        text = tokenizer.encode(query).to(device)

        for x, y in loader:
            x = x.to(device)
            image_pred, text_pred = model(x, text, return_loss=False)
            predictions += image_pred.argmax(dim=-1).cpu().data.numpy().tolist()
            ground_truths += y.data.numpy().tolist()

    import pdb; pdb.set_trace()
    return predictions, ground_truths



if __name__ == '__main__':
    get_imagenet()
