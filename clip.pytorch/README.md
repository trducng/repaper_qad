# Play with OpenAI's CLIP

Code taken and modified from https://github.com/weiyx16/CLIP-pytorch.  Not
necessary to do a training for CLIP at this moment. It is trained on 400
millions image-text pairs for 32 epochs, which would cost several thousands of
GPU just to train for a hundred days. Instead, we should focus on answering
these questions:

1. Where does the VisionTransformer attend to given a text prompt?
2. What would happen if we change the text prompt given the same input image?
3. What would happen if we change the text prompt given the same input image
   where the image contains multiple objects?
4. How does changing the prompt alter the encoded text projection?
5. How does changing the prompt alter the combination space of text + image
   embedding?

Hypothesis: the CLIP model contains wide variety of general concepts, because
the image-text pairs are retrieved en-mass from the Internet so the model is
familiar with the concepts.

Current replacement of convolution -> transformer seems to be very primitive. It
cannot deal with variable resolution. It has rather wide receptive field. It
also seems to have unnecessary high dimensions in the lower layer.

1. Focus on the supporting attention to the CLS token -> 1st row of attention map
(try attention roll-out method suggested in D.7 in Vision Transformer paper).
This method does not take into account the text, so it might not show much and
we need to come up with another way to show the attention depending on the text
prompt.


In multi-head attention, each head only has 64 dimensions which means it does
not need to be very high dimension.
