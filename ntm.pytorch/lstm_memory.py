"""LSTM cell with attached memory (implemented in Pytorch)
@author: _john

The memory is not essential component of LSTM. In that case, it might be
possible to 'attach' the memory inside LSTM. If we do that way, multiple layers
of LSTM can access a single memory component.

The LSTM should interact with the memory in every input timestep. Current
Pytorch LSTM implementation takes all timesteps as input, and then return
all output along with last hidden state and cell state. For this reason,
LSTMCell, GRUCell might be a better choice interaction with memory.

The reason Pytorch's LSTM wants to consume the whole input is because it
supports bidirectional operations: it calculates both forward and backward, in
a single pass, so it needs to know the sequence beforehand. As a result, the
memory version of LSTM should subclass LSTMCell, instead of LSTM.

Note: it seems to learn, however the testing is not rigorous, as the model
basically has to output the same input in each timestep, hence the role of
memory cannot be clearly evaluated (the input at each timestep is indepedent
from each other).Maybe doing a delayed timestep?
"""
import pdb
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LSTMNTM_STATE = namedtuple(
    'LSTMNTM_STATE',
    ('rnn_hidden_state', 'rnn_cell_state', 'read_attentions',
     'write_attentions', 'memory'))


class LSTM_NTM(nn.RNNBase):
    """Equip memory per NTM to LSTM implementation

    # Arguments
        controller [nn.RNNBase]: use RNNBase method
        memory_size [int]: the number of memory slots

    """

    def __init__(self, input_size, hidden_size, memory_size, memory_dim,
                 n_read_heads, n_write_heads, shift, num_layers=1, bias=True,
                 batch_first=False, dropout=0):
        """Initialize the object"""
        input_size_ = input_size + memory_dim * n_read_heads
        super(LSTM_NTM, self).__init__(
            mode='LSTM', input_size=input_size_, hidden_size=hidden_size,
            num_layers=num_layers, bias=bias, batch_first=batch_first,
            dropout=dropout, bidirectional=False)

        self.timestep_dim = 1 if self.batch_first else 0

        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.shift = shift

        # @NOTE: if memory is created this way, then it can also be added, in
        # way that multiple LSTMs or GRUs access a same block of memory
        self.memory = self._initialize_memory_block()
        self.read_attentions = [
            self._initialize_head()
            for _ in range(self.n_read_heads)]
        self.write_attentions = [
            self._initialize_head()
            for _ in range(self.n_write_heads)]

        self.params_per_head = self.memory_dim + 1 + 1 + 1 + (self.shift * 2 + 1)
        self.controller_to_interface = nn.Linear(
            in_features=hidden_size,
            out_features=self.params_per_head
            * (self.n_read_heads + self.n_write_heads)
            + self.n_write_heads * 2 * self.memory_dim
        )

    def forward(self, input, hx=None):
        """Subclass the forward method from nn.RNNBase to incorporate the use
        of memory

        # Arguments
            input [Tensor]: the input to the model, typically of shape
                [timestep x batchsize x dim]
            hx [Tensor]: the LSTM hidden state

        # Returns
            [Tensor]: the output
            [Tensor]: the hidden state
        """
        batch_size = input.size(1 - self.timestep_dim)
        if hx is not None:
            hidden_state = (hx.rnn_hidden_state, hx.rnn_cell_state)
            self.memory = hx.memory
            self.read_attentions = hx.read_attentions
            self.write_attentions = hx.write_attentions
        else:
            hidden_state = None
            self.memory = self._initialize_memory_block()
            self.read_attentions = [
                self._initialize_head().repeat(batch_size, 1).unsqueeze(-1)
                for _ in range(self.n_read_heads)]
            self.write_attentions = [
                self._initialize_head().repeat(batch_size, 1).unsqueeze(-1)
                for _ in range(self.n_write_heads)]

        # expand the memory to accommodate for batch size
        # memory of size [batch size x height x width]
        memory = self.memory.repeat(batch_size, 1, 1)

        rnn_hidden_state, rnn_cell_state = None, None
        outputs = []
        for timestep in range(input.size(self.timestep_dim)):
            timestep_input = input[:, timestep, :] if self.batch_first else input[timestep]     # B x I_dim
            read_vectors = [(self.memory * each_vector).sum(1)
                            for each_vector in self.read_attentions]
            # each read_vector is supposed to be a matrix of size (B x M_w)

            # get the controller temporary output
            controller_input = torch.cat(
                [timestep_input] + read_vectors, dim=1).unsqueeze(0)
            controller_output, hidden_state = super(LSTM_NTM, self).forward(
                controller_input, hidden_state)
            controller_output = controller_output.squeeze() # [B x hidden_size]
            if controller_output.dim() < 2:
                controller_output = controller_output.unsqueeze(0)
            rnn_hidden_state, rnn_cell_state = hidden_state

            # from the controller temporary output, get the interface vectors
            interface = self.controller_to_interface(controller_output)
            heads = torch.split(
                interface[:, :self.params_per_head
                          * (self.n_read_heads + self.n_write_heads)],
                self.params_per_head,
                dim=1)
            erase_add = torch.split(
                interface[:, self.params_per_head
                          * (self.n_read_heads + self.n_write_heads):],
                self.memory_dim,
                dim=1)

            # interface vector -> attention weights
            prev_attentions = self.read_attentions + self.write_attentions
            attentions = []
            for idx, each_head in enumerate(heads):
                # self.memory_dim + 1 + 1 + 1 + (self.shift * 2 + 1)
                k = torch.tanh(each_head[:, :self.memory_dim])
                beta = F.softplus(each_head[:, self.memory_dim])
                g = torch.sigmoid(each_head[:, self.memory_dim + 1])
                gamma = F.softplus(each_head[:, self.memory_dim + 2])
                s = torch.softmax(each_head[:, self.memory_dim + 3:], dim=1)

                attentions.append(self.get_attention_weight(
                    k, beta, g, s, gamma, memory, prev_attentions[idx]).unsqueeze(-1))    # [B x W_h x 1]
            self.read_attentions = attentions[:self.n_read_heads]
            self.write_attentions = attentions[self.n_read_heads:]

            # retrieve the read vectors
            read_vectors = [(self.memory * each_vector).sum(1)      # [B x W_w]
                            for each_vector in self.read_attentions]

            # modify memory
            for idx, each_write in enumerate(self.write_attentions):
                erase = torch.sigmoid(erase_add[idx * 2]).unsqueeze(1)   # [B x 1 x W_w]
                add = torch.tanh(erase_add[idx * 2 + 1]).unsqueeze(1)    # [B x 1 x W_w]
                memory = memory - memory * erase * each_write   # [B x W_h x W_w]
                self.memory = memory + add * each_write
            
            outputs.append(torch.cat([controller_output] + read_vectors, dim=1))    # [B x hidden_size + n_read * W_w]

        # concat the controller temporary output with the read head and get
        # the real output that satisfies the output dimension
        # @NOTE: this step might be unnecessary since the way to transform from
        # read-conditioned output to desired output depends on whatever model
        # builder want

        return torch.stack(outputs), LSTMNTM_STATE(
            rnn_hidden_state=rnn_hidden_state, rnn_cell_state=rnn_cell_state,
            read_attentions=self.read_attentions,
            write_attentions=self.write_attentions,
            memory=self.memory)

    def get_attention_weight(self, k, beta, g, s, gamma, mem, prev_attention):
        """Get the attention weight from memory

        # Arguments
            k [Tensor]: controller content [B x M_w]
            beta [Tensor]: softmax beta [B]
            g [Tensor]: interpolated value [B]
            s [Tensor]: attention shift [B x 2*self.shift+1]
            gamma [Tensor]: gamma smoothing [B]
            mem [Tensor]: memory object [B x M_h x M_w]
            prev_attention [Tensor]: previous head attention weights [B x M_h x 1]
        """
        # Content-based retrieval using cosine similarity
        # k = k.unsqueeze(-1).contiguous()        # B x M_w x 1
        # inner_product = torch.matmul(mem, k)    # batch multiplication, B x M_h x 1
        # # k_norm B x 1 x 1, mem_norm B x M_h x 1
        # k_norm = torch.sqrt(torch.sum(k ** 2, axis=1, keepdim=True))
        # mem_norm = torch.sqrt(torch.sum(mem ** 2, axis=2, keepdim=True))
        # cosine_similarity = inner_product / (k_norm * mem_norm + 1e-8)  # B x M_h x 1
        # cosine_similarity = cosine_similarity.squeeze()     # B x M_h
        k = k.unsqueeze(1).contiguous()         # B x 1 x M_w
        cosine_similarity = F.cosine_similarity(mem, k, dim=2)  # B x M_h

        # softmax-like operation
        # cosine_similarity = torch.exp(cosine_similarity * beta.unsqueeze(-1))
        # w_c = cosine_similarity / cosine_similarity.sum(axis=1, keepdim=True)
        w_c = F.softmax(cosine_similarity * beta.unsqueeze(-1), dim=1)

        # Interpolate with the previous attention result
        g = g.unsqueeze(-1)     # B x 1
        w_g = g * w_c + (1 - g) * prev_attention.squeeze()    # B x W_h

        # Location-based retrieval using attention shift
        output_ = []
        for each_item in range(w_g.size(0)):
            # w_g_ [W_h], s_ [2 * self.shift + 1]
            w_g_, s_ = w_g[each_item], s[each_item]
            w_g_ = torch.cat([w_g_[-self.shift:], w_g_, w_g_[:self.shift]])
            output_.append(F.conv1d(w_g_.view(1, 1, -1), s_.view(1, 1, -1)).view(-1))     # [W_h]
        w_g = torch.stack(output_)                      # [B x W_h]
        w_g = w_g ** gamma.unsqueeze(-1)
        w_g = w_g / w_g.sum(dim=1, keepdim=True) + 1e-8

        return w_g

    def _initialize_memory_block(self, memory_size=None, memory_dim=None):
        """Initialize a memory block

        # Arguments
            memory_size [int]: number of memory slots
            memory_dim [int]: the dimension of each memory slot

        # Returns
            [Tensor]: the memory block of shape [memory_size x memory_dim]
        """
        memory_size = self.memory_size if memory_size is None else memory_size
        memory_dim = self.memory_dim if memory_dim is None else memory_dim

        return torch.ones([memory_size, memory_dim], dtype=torch.float32) * 1e-6

    def _initialize_head(self, memory_size=None):
        """Initialize a read/write head

        # Arguments
            memory_size [int]: the number of memory slots

        # Returns
            [1D Tensor]: the attention weights of size (memory_size,)
        """
        memory_size = self.memory_size if memory_size is None else memory_size

        return torch.softmax(torch.randn([memory_size], dtype=torch.float32), dim=0)


if __name__ == '__main__':
    try:
        import toys
    except ModuleNotFoundError:
        print('Please run `pip install toys`')
        exit(0)

    import numpy as np
    import torch.optim as optim

    IN_FEATURES = 5

    class Model(nn.Module):
        """Simple model"""
        def __init__(self, input_size, output_size):
            """Initialize the model"""
            super(Model, self).__init__()

            self.input_size = input_size
            self.lstm = LSTM_NTM(
                input_size=input_size, hidden_size=6, num_layers=2,
                memory_size=10, memory_dim=6, n_read_heads=2, n_write_heads=3,
                shift=1)
            self.output = nn.Linear(in_features=18, out_features=output_size)

            self.hidden_state = None

        def forward(self, input_, timestep=None):
            """Run the model on the input"""
            if input_ is None:
                input_ = torch.zeros(timestep, 1, self.input_size)
            else:
                self.hidden_state = None

            hidden, self.hidden_state = self.lstm(input_, self.hidden_state)
            hidden = self.output(hidden)

            return torch.sigmoid(hidden)

    task = toys.Copy(IN_FEATURES, max_timestep=10)
    model = Model(IN_FEATURES + 1, IN_FEATURES)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9)

    idx = 0
    while True:
        idx += 1

        in_batch, out_batch, widths = task.get_batch(1)
        in_batch = torch.Tensor(in_batch).transpose(0, 1).contiguous()
        out_batch = torch.Tensor(out_batch).transpose(0, 1).contiguous()

        _ = model(in_batch)
        logits = model(input_=None, timestep=out_batch.size(0))

        loss = criterion(logits, out_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Idx: {} - Loss: {} - Seq length: {}'.format(
            idx, loss.item(), widths[0]))

        if idx % 10 == 0:
            out_batch = out_batch.squeeze().data.numpy().transpose()
            logits = logits.squeeze().data.numpy().transpose()
            logits = (logits >= 0.5).astype(np.uint8)
            diff = (logits == out_batch).astype(np.uint8)

            out_batch = np.where(out_batch >= 0.5, '+', '-')
            logits = np.where(logits >= 0.5, '+', '-')
            diff = np.where(diff < 0.5, '+', '-')

            out_batch = [''.join(list(each)) for each in out_batch]
            logits = [''.join(list(each)) for each in logits]
            diff = [''.join(list(each)) for each in diff]

            to_print = ['{}   {}   {}'.format(
                out_batch[idx], logits[idx], diff[idx])
                for idx in range(len(out_batch))]

            print('')
            print('Batch - Logits - Diff')
            print('\n'.join(to_print))
            print()
