from CLIP import CLIP, build_transform
from clip_tokenizer import SimpleTokenizer
from PIL import Image
import torch
from torchvision import transforms


def get_attention_maps(model, visual=True):
    if visual:
        component = model.visual.transformer
    else:   # for text
        component = model.transformer

    attention_layers = []
    for layer in component.resblocks._modules.values():
        attention_layers.append(layer.attention_weights)

    attention_layers = torch.stack(attention_layers, dim=0) # layers x head x t x t
    attention_layers = torch.mean(attention_layers, dim=1)  # layers x t x t
    res_attention = torch.eye(attention_layers.size(1))
    attention_layers += res_attention
    attention_layers /= attention_layers.sum(dim=-1, keepdim=True)

    final = torch.zeros(attention_layers.size())
    final[0] = attention_layers[0]
    for idx in range(1, final.size(0)):
        final[idx] = torch.matmul(attention_layers[idx], final[idx-1])

    return final


if __name__ == '__main__':

    MODEL_PATH = 'clip.pth'
    VOCAB_PATH = 'bpe_simple_vocab_16e6.txt.gz'

    model = CLIP(attention_probs_dropout_prob=0, hidden_dropout_prob=0)
    model.load_state_dict(state_dict = torch.load(MODEL_PATH))

    tokenizer = SimpleTokenizer(
            bpe_path=VOCAB_PATH,
            context_length=model.context_length.item())
    transform = build_transform(model.input_resolution.item())
    view_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert('RGB')])
    is_fp16 = False

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    if is_fp16:
        model.to(device=device).eval().half()
    else:
        model.to(device=device).eval().float()

    with torch.no_grad():
        query = ["a balo", "a human", "a tiger", "a cat", "a human and a tiger"]
        text = tokenizer.encode(query).to(device)
        text_features = model.encode_text(text)  # N_queries x 512

        image_path = "images/balloon.jpg"
        # view_transform(Image.open(image_path)).show()
        image = transform(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image) # 1 x 512

        text_attention = get_attention_maps(model, visual=False)
        visual_attention = get_attention_maps(model, visual=True)

        import pdb; pdb.set_trace()

        logits_per_image, logits_per_text = model(image, text, return_loss=False)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)

