import torch, torch.nn as nn


"""
A Seq2Seq LSTM model, with options for bidirectionality, Luong, Bahdanau, or no Attention mechanism and different normalisation layers. 
"""

# encoder decoder structure follows https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html and supervisor code
# NOTE: allows for n_layers > 1 but doesn't necessary work with it
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,  bidirectional, normalisation, n_layers = 1, dropout_p = 0.05,) -> None:
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.normalisation = normalisation
        self.n_layers = n_layers
        # turn input into a hidden_size sized vector, use to try and learn relationship between pitches and FB notations 
        self.embedding = nn.Embedding(input_size, hidden_size)

        self.rnn = nn.LSTM(hidden_size, hidden_size,num_layers= n_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer_norm_hn = nn.LayerNorm(hidden_size)
        self.layer_norm_cn = nn.LayerNorm(hidden_size)
        # self.layer_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        if self.normalisation == "dropout" or self.normalisation == "both":
            embedded = self.dropout(embedded)

        output, hidden = self.rnn(embedded)

        if (self.bidirectional):
            # sum forward and backwards output and hidden states together
            # to work with Attention
            output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]

            hidden_layers = []
            cell_layers = []
            for i in range(self.n_layers):
                # add forwards and backwards state in pairs
                (h_n, c_n) = hidden
                concat_h = (h_n[i*2, :, :] + h_n[i*2 + 1,:,:]).unsqueeze(0)
                concat_c = (c_n[i*2, :, :] + c_n[i*2 + 1,:,:]).unsqueeze(0)

                hidden_layers.append(concat_h)
                cell_layers.append(concat_c)
            
            hidden = (torch.cat(hidden_layers,0), torch.cat(cell_layers, 0))

        # layer normalisation
        if self.normalisation == "layer" or self.normalisation == "both":
            output = self.layer_norm(output)
            hidden[0] = self.layer_norm_hn(hidden[0])
            hidden[1] = self.layer_norm_hn(hidden[1])

            # output = self.layer_norm(output.permute(0,2,1))
            # output = output.permute(0,2,1)

        return output, hidden
    
    # check embedding
    def get_embedding(self, x):
        return self.embedding(x)

# Attention module based off
# https://github.com/rawmarshmellows/pytorch-batch-luong-attention/blob/master/models/luong_attention/luong_attention.py
# General Luong attention
class LuongAttention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, decoder_output, encoder_outputs):
        # for us, decoder output = batch size, 1, hidden size
        # encoder_output = batch size, max input (default 3), hidden size
        # change to = batch size, hidden, length

        attn_energies = torch.bmm(self.attn(decoder_output), encoder_outputs.permute(0, 2, 1))


        # Batch size, 1, max input length
        return nn.functional.softmax(attn_energies, dim = -1)

    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output).view(-1)
        energy = hidden.view(-1).dot(energy)
        return energy

    
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#attention-decoder
# bahdanau attention based off above
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    # query = last decoder hidden state, (batch, layers, hidden size)
    # keys = set of encoder outputs
    def forward(self, query, keys):

        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class DecoderRNN(nn.Module):

    # output size typically equals encoder's input size
    # NOTE: "timesteps" are columns of output.
    def __init__(self,  hidden_size, output_size, attention_model, device, SOS_token, normalisation, output_num = 6,  n_layers =1, dropout_p = 0.1) -> None:
        super(DecoderRNN, self).__init__()


        self.output_num = output_num
        self.n_layers = n_layers
        self.device = device
        self.SOS_token = SOS_token
        self.normalisation = normalisation

        self.embedding = nn.Embedding(output_size, hidden_size)
        if attention_model == "bahdanau":
            self.rnn = nn.LSTM(hidden_size*2, hidden_size, n_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer_norm_hn = nn.LayerNorm(hidden_size)
        self.layer_norm_cn = nn.LayerNorm(hidden_size)
        # self.layer_norm = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        if attention_model == "luong":
            self.attention = LuongAttention(hidden_size)
            self.concat = nn.Linear(2*hidden_size, hidden_size)
        elif attention_model == "bahdanau":
            self.attention = BahdanauAttention(hidden_size)

        self.attention_model = attention_model
        self.linear = nn.Linear(hidden_size*n_layers, output_size)
        

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        batch_size = encoder_outputs.size(0)

        # second dimension = 1 - means we are processing 1 timestep at a time
        decoder_input = torch.empty(batch_size, self.n_layers, dtype=torch.long, device=self.device).fill_(self.SOS_token)

        # context vector from encoder used as initial hidden state
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        # process num_outputs columns at a time - typically, the 6 values we want
        # for the whole of the batch
        for i in range(self.output_num):
            decoder_output, decoder_hidden, attention_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)

            decoder_outputs.append(decoder_output)
            attentions.append(attention_weights)

            if target_tensor is not None:
                # feed target as next input/column 
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # use own prediction as next input 
                # use whilst generating/predicting - we don't want model to know true values
                vals, indices = decoder_output.topk(1)
                decoder_input = indices.squeeze(-1).detach()


        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        if self.attention_model is not None:
            attentions = torch.cat(attentions, dim = 1)
            return decoder_outputs, decoder_hidden, attentions
        
        else:
            return decoder_outputs, decoder_hidden, None
    
    # for a single input - embed, rnn, use linear output layer
    def forward_step(self, input, hidden, encoder_outputs):
        output = self.embedding(input)
        if hasattr(self, "attention") and (self.normalisation == "dropout" or self.normalisation == "both"):
            output = self.dropout(output)


        if self.attention_model == "bahdanau":
            # since hidden is a tuple
            # length, batch, hidden -> batch, length, hidden
            query = hidden[0].permute(1,0,2)

            context, attn_weights = self.attention(query, encoder_outputs)
            output = torch.cat((output, context), dim = 2)

        output, hidden = self.rnn(output, hidden)
        #normalisation
        if self.normalisation == "layer" or self.normalisation == "both":
            output = self.layer_norm(output)
            hidden[0] = self.layer_norm_hn(hidden[0])
            hidden[1] = self.layer_norm_hn(hidden[1])
            # output = self.layer_norm(output.permute(0,2,1))
            # output = output.permute(0,2,1)


        if self.attention_model == "luong":
            # attention weight calculations from current rnn output
            weights:torch.Tensor = self.attention(output, encoder_outputs)

            # new weighted sum context = weights * encoder outputs
            # [batch_size, 1, max input length] @ [batch_size, max input length, hidden size]
            context = weights.bmm(encoder_outputs).squeeze(1)

            # remove sequence length at index 1, since batch_first = true
            output = output.squeeze(1)

            concat_input = torch.cat((output, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))

            # predict next token
            # unsqueeze to get [batch size, seq length, hidden size] format
            final_output = self.linear(concat_output).unsqueeze(1)
        elif self.attention_model == "bahdanau":
            weights = attn_weights
            final_output = self.linear(output)
        else:
            weights = None
            final_output = self.linear(output)

        return final_output, hidden, weights

    def get_embedding(self, x):
        return self.embedding(x)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        encode_out, encode_hid = self.encoder(x)
        decode_out, decode_hid, attention_weights = self.decoder(encode_out, encode_hid, y)
        return decode_out, decode_hid, attention_weights
