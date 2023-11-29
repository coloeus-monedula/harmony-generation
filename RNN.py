import argparse
import muspy
from manual_harmony import convert_music21
import torch.nn as nn
import torch
from local_datasets import PytorchChoralesDataset as Chorales, PytorchSplitChoralesDataset as SplitChorales
from torch.utils.data import DataLoader, TensorDataset
import time, math
from tokeniser import Tokeniser
import dill as pickle

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


# https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-rnn-cb6ebc594677 
# https://aizardar.github.io/blogs/pytorch/classification/rnn/2021/01/07/seq-classification.html 

# https://qr.ae/pKklS7 re hidden layers - try one for now

# input size = 3, output size = 6
# using 3 features - s,acc,fb so input size = 3. output size = 6

# TODO: stacked RNN? t-1, t, t+1
# hidden dim is hyperparam, find outside of model

# TODO: currently not doing batch numbers, just input one whole piece at a time
# TODO: if want to use batch sizes, break up the scores?? probably want each batch to be a score due to each piece being "separate" 
#  use padding like in floydhub if batching

# encoder decoder structure follows https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html and supervisor code
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1, dropout_p = 0.05) -> None:
        super(EncoderRNN, self).__init__()

        # turn input into a hidden_size sized vector, use to try and learn relationship between pitches and FB notations 
        self.embedding = nn.Embedding(input_size, hidden_size)

        # TODO: change to lstm, keep in mind also has c_n output
        self.rnn = nn.LSTM(hidden_size, hidden_size,num_layers= n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded)

        return output, hidden
    
    # check embedding
    def get_embedding(self, x):
        return self.embedding(x)


class DecoderRNN(nn.Module):

    # output size typically equals encoder's input size
    def __init__(self,  hidden_size, output_size, output_num = 6,  n_layers =1) -> None:
        super(DecoderRNN, self).__init__()

        self.output_num = output_num
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        batch_size = encoder_outputs.size(0)

        # second dimension = 1 - means we are processing 1 timestep at a time
        decoder_input = torch.empty(batch_size, self.n_layers, dtype=torch.long, device=device).fill_(parameters["SOS_TOKEN"])
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        # process num_outputs timesteps/columns at a time - typically, the 6 values we want
        # for the whole of the batch
        for i in range(self.output_num):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # feed target as next input/column 
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # use own prediction as next input 
                vals, indices = decoder_output.topk(1)
                decoder_input = indices.squeeze(-1).detach()


        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = nn.functional.log_softmax(decoder_outputs, dim = -1)

        return decoder_outputs, decoder_hidden
    
    # for a single input - embed, rnn, use linear output layer
    def forward_step(self, input, hidden):
        output = self.embedding(input)
        # NOTE: need to add relu? don't think so
        output, hidden = self.rnn(output, hidden)
        output = self.linear(output)

        return output, hidden

    def get_embedding(self, x):
        return self.embedding(x)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        encode_out, encode_hid = self.encoder(x)
        decode_out, decode_hid = self.decoder(encode_out, encode_hid, y)
        return decode_out, decode_hid

def time_since(since):
    now = time.time()
    secs = now-since
    mins = math.floor(secs/60)
    secs -= mins * 60
    return '%dmins %dsecs' % (mins, secs)

def train(model:EncoderDecoder, loader:DataLoader, criterion:nn.CrossEntropyLoss, optimiser:torch.optim.Adam, hyperparameters):

    all_losses = []
    start = time.time()

    for epoch in range(hyperparameters["n_epochs"]):
        total_loss = 0
        # if batches = 1 then num_batches = number of scores
        # num_batches = len(loader)
        # number of rounds
        num_rounds = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            output, hidden =  model(x, y)

            # print(decode_out.size())
            predicted = output.reshape(hyperparameters["batch_size"] * hyperparameters["output_num"], -1)
            # flattens 
            flattened_y = y.reshape(-1)
            loss =criterion(predicted, flattened_y)

            # backwards propagation
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            total_loss += loss.item()
            num_rounds+=1

        
        if epoch % hyperparameters["print_every"] == 0:
            print('Epoch: {}/{},'.format(epoch+1, hyperparameters["n_epochs"]), end=" ")
            print(time_since(start),end = "......................... ")
            print("Loss: {:4f}".format(total_loss/num_rounds))

        # add average loss per epoch across all batches to plot graph
        if epoch % hyperparameters["plot_every"] == 0:
            all_losses.append(total_loss/num_rounds)

    return all_losses


# splits scores into individual timesteps to use with batch processing
def split_scores(dataset: SplitChorales) -> TensorDataset:
    all_x, all_y = [], []

    # x and y
    for i in range(len(dataset)):
        all_x.extend(dataset[i][0])
        all_y.extend(dataset[i][1])

    all_x = torch.stack(all_x).long()
    all_y = torch.stack(all_y).long()

    return TensorDataset(all_x, all_y)

# pad end of batch
def pad(split_tensors):
    modulo = len(split_tensors) % parameters["batch_size"]
    to_pad = parameters["batch_size"] - modulo
    total_padding_x = []
    total_padding_y = []
    for i in range(to_pad):
        total_padding_x.append(torch.tensor([0,0,0]).long())
        total_padding_y.append(torch.tensor([0,0,0,0,0,0]).long())

    total_padding_x = torch.stack(total_padding_x).long()
    total_padding_y = torch.stack(total_padding_y).long()
    
    total_padding = TensorDataset(total_padding_x, total_padding_y)
    split_tensors = split_tensors + total_padding
    return split_tensors




# hyperparams and params
parameters = {
    "lr": 0.01,
    "n_epochs": 100,
    # measured in epoch numbers
    "plot_every" : 5,
    "print_every" : 10,
    "batch_size": 32,
    "hidden_size": 128,
    #the unknown token is set as 250 and if you set input size = unknown token num it gives an out of index error when reached
    # # possibly because 0 is also used as a token so off by 1
    # "input_size" : 252, 
    "output_num": 6,
    "SOS_TOKEN": 129 #for the decoder
}

tokens = Tokeniser()

def main():
    # parser = argparse.ArgumentParser()

    # # input size should be = 3

    # args = parser.parse_args()
    eval = False

    path = "content/model.pt"
    token_path = "content/tokens.pkl"
    split = True
    file = "content/preprocessed.pt"


    if split:
        dataset = SplitChorales(file)
    else:
        dataset = Chorales(file)

    split_tensors = split_scores(dataset)
    split_tensors = pad(split_tensors)

    # shuffle = false since data is time contiguous + to learn when an end of piece is
    loader = DataLoader(split_tensors, batch_size=parameters["batch_size"], shuffle=False)

    # calculate input size dynamically by the tokeniser
    # add 1 since 0 is also used as a token to avoid out of index errors
    with open(token_path, "rb") as f:
        tokens.load(pickle.load(f))

    input_size = tokens.get_max_token() + 1
    hidden_size = parameters["hidden_size"]
    output_num = parameters["output_num"]

    encoder = EncoderRNN(input_size, hidden_size)
    decoder = DecoderRNN(hidden_size, input_size, output_num=output_num)
    encode_decode = EncoderDecoder(encoder, decoder)

    # loss and optimiser
    optimiser = torch.optim.Adam(encode_decode.parameters(), lr = parameters["lr"])

    if eval:
        print("Evaluating model.")

        checkpoint = torch.load(path)
        # encoder.load_state_dict(checkpoint["encode"])
        # decoder.load_state_dict(checkpoint["decode"])
        encode_decode.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])

        encode_decode.eval()

    else:
        print("Training model.")
        criterion = nn.CrossEntropyLoss()

        encoder.to(device)
        decoder.to(device)
        train(encode_decode, loader, criterion, optimiser, parameters)

        # save model
        torch.save({
            "model": encode_decode.state_dict(),
            "optimiser": optimiser.state_dict(),
        }, path)






if __name__ == "__main__":
    main()