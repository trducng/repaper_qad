"""Vanilla Sequence-to-Sequence with attention architecture"""
import random

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from features import get_densenet_3blocks


class Encoder(nn.Module):
    """Feature extractor
    
    # Arguments
        hidden_size [int]: the number of neurons in hidden feature vector
        bidirectional [bool]: whether the built-in RNN is bi-directional
    """

    def __init__(self, hidden_size, bidirectional): 
        """Initialize the object"""
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # we will make use of stock DenseNet as feature extractor
        self.feature, channels, height = get_densenet_3blocks()

        self.brnn = nn.LSTM(input_size=channels * height,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional)
        self.output = nn.Linear(in_features=hidden_size * (bidirectional + 1),
                                out_features=hidden_size)

    def forward(self, input_tensor, length=None):
        """Perform the forward pass
        
        # Arguments
            input_tensor [4d tensor]: shape (B x C x H x W)

        # Returns
            [3d tensor]: shape (B x C x hidden)
        """
        batch_size = input_tensor.size(0)
        
        features = self.feature(input_tensor)       # B x C x H1 x W1
        features = features.permute(3, 0, 1, 2)     # W1 x B x C x H1
                                                    # W1 x B x U
        features = features.view(features.size(0), features.size(1), -1)
        features, _ = self.brnn(features)           # W1 x B x hidden*(bi+1)
        output = F.relu(self.output(features))      # W1 x B x hidden

        return output


class AttentionalDecoder(nn.Module):
    """The decoder for Seq2Seq, that already incorporates Attention
    
    # Arguments
        embedding_size [int]: the size of the character embedding
        hidden_size [int]: the size of the hidden network
        output_size [int]: the number of classes
        dropout [float]: the dropout probability
    """
    
    def __init__(self, embedding_size, hidden_size, output_size, dropout_p):
        """Initialize the architecture"""
        super(AttentionalDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.content_attention = nn.Linear(hidden_size, 1)
        self.combine = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, last_input, last_hidden, encoder_outputs):
        """Perform the forward pass
        
        # Arguments
            last_output [int]: the predicted character of last timestep (it
                can be the true character of the last timestep in case of
                teacher forcing)
            last_hidden [3D tensor]: the hidden state of the last timestep,
                shape [1 x B x output_size]
            encoder_outputs [3D tensor]: the encoder output, should have shape
                T x B x hidden

        # Returns
            [2D tensor]: the pre-softmax activation [B x output_size]
            [tuple of 2 3D tensor]: the hidden and cell state of rnn
            [2D tensor]: the attention, of shape [B x T]
        """
        batch_size = encoder_outputs.size(1)

        # transform the last input
        embedded = self.embedding(last_input)
        embedded = self.dropout(embedded)

        # combine the last hidden state with the encoder result to understand
        # past knowledge
        past_knowledge = last_hidden + encoder_outputs      # T x B x hidden
        past_knowledge = torch.tanh(past_knowledge)

        # get attention weights
        attention_weights = self.content_attention(past_knowledge)  # T x B x 1
        attention_weights = F.softmax(attention_weights, dim=0)     # T x B x 1

        # apply attention
        attended_output = torch.matmul(
            attention_weights.permute((1, 2, 0)),           # B x 1 x T
            encoder_outputs.permute((1, 0, 2))              # B x T x hidden
        )                                                   # B x 1 x hidden
                                                        # B x (hidden + embed) 
        output = torch.cat((embedded, attended_output.squeeze(1)), 1)
        output = self.combine(output).unsqueeze(0)      # 1 x B x hidden

        output = F.relu(output) 
        
        output, hidden = self.gru(output, last_hidden)
        output = self.out(output[0])

        return output, hidden, attention_weights

    def initialize_hidden_state(self, batch_size):
        """Initialize hidden state of the RNN"""

        return torch.zeros(1, batch_size, self.hidden_size)


class Seq2Seq(nn.Module):
    """Vanilla sequence-to-sequence implementaiton
    
    # Arguments
        embedding_size [int]: the size of the character embedding
        hidden_size [int]: the size of the hidden network
        output_size [int]: the number of classes
        bidirectional [bool]: whether using bi-rnn in the encoder
        dropout [float]: the dropout probability
    """

    def __init__(self, embedding_size, hidden_size, output_size,
                 bidirectional=True, dropout_p=0.1):
        """Initialize the object"""
        super(Seq2Seq, self).__init__()

        self.output_size = output_size
        self.encoder = Encoder(hidden_size=hidden_size,
                               bidirectional=bidirectional)
        self.decoder = AttentionalDecoder(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout_p=dropout_p
        )
        
    def forward(self, input_tensor, input_label, teacher_prob=0.5):
        """Perform the forward pass

        # Arguments
            input_tensor [4D tensor]: the images of shape [B x C x H x W]
            input_label [2D tensor]: the label of shape [B x max_char]
            teacher_prob [float]: probability of doing teacher forcing when
                the input label is provided

        # Returns
            [3D tensor]: model's pre-softmax distribution 
            [3D tensor]: the attention of shape [T x B x T]
        """
        mini_batch = input_tensor.size(0)

        encoder_output = self.encoder(input_tensor)
        output = input_label[:,0]
        hidden = self.decoder.initialize_hidden_state(mini_batch)
        first_prediction = torch.zeros(mini_batch, self.output_size).float()
        first_prediction[:, 0] = 1.0

        hidden = hidden.cuda()
        first_prediction = first_prediction.cuda()

        outputs, attentions = [first_prediction], []
        for each_input in range(1, input_label.size(1)):
            output, hidden, attention_weights = self.decoder(
                output, hidden, encoder_output)

            outputs.append(output)
            attentions.append(attention_weights)

            if random.random() < teacher_prob:
                output = input_label[:, each_input]
            else:
                output = torch.argmax(output, dim=1)
        
        outputs = torch.stack(outputs, dim=0)                   # T x B x C
        outputs = outputs.permute((1, 2, 0))                    # B x C x T

        return outputs, attentions

