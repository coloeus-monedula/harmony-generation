import argparse
import muspy
from manual_harmony import convert_music21
import torch.nn as nn
import torch
from local_datasets import PytorchChoralesDataset as Chorales, PytorchSplitChoralesDataset as SplitChorales
from torch.utils.data import DataLoader, TensorDataset
import time, math

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
    def __init__(self, input_size, hidden_size, n_layers = 1, dropout_p = 0.1) -> None:
        super(EncoderRNN, self).__init__()

        # turn input into a hidden_size sized vector, use to try and learn relationship between pitches and FB notations 
        self.embedding = nn.Embedding(input_size, hidden_size)

        # TODO: change to lstm, keep in mind also has c_n output
        self.rnn = nn.RNN(hidden_size, hidden_size,num_layers= n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        print("Input shape:", input.shape)
        print("Embedding layer params:", self.embedding.weight.shape)
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
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        batch_size = encoder_outputs.size(0)

        # zeroes should be fine for us since 0 is used as silence
        # second dimension = 1 - means we are processing 1 timestep at a time
        decoder_input = torch.zeros(batch_size, self.n_layers, dtype=torch.long, device=device)
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


# follows https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers=1) -> None:
        super(LSTM, self).__init__()

        # dimensions for hidden layers
        self.hidden_size = hidden_size
        # number of RNN layers
        self.n_layers = n_layers

        # will be made up of n_layers of LSTM
        # DataLoader does batch number first so switch to batch first
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        # reshape to output 
        self.linear = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        batch_size = x.size(0)
        h_n = self.init_hidden(batch_size)
        

        # final hidden state for each element in batch
        output, h_n = self.lstm(x, h_n)

        # make output stored in same block of memory
        # fit to hidden_size dimensiions to pass through linear layer
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.linear(output)

        # add extra (batch) dimension to output??
        # output = output[None,:,:]

        return output, h_n


    def init_hidden(self, batch_size):
        # initialise the hidden state - makes a layers x batch size x hidden size sized tensor 
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.long)




def time_since(since):
    now = time.time()
    secs = now-since
    mins = math.floor(secs/60)
    secs -= mins * 60
    return '%dmins %dsecs' % (mins, secs)

def train(encoder:EncoderRNN, decoder:DecoderRNN, loader:DataLoader, criterion:nn.NLLLoss, e_optimiser:torch.optim.Adam, d_optimiser:torch.optim.Adam, hyperparameters):

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

            print(x.size())
            try:
                encode_out, encode_hid = encoder(x)
            except IndexError:
                print(x)
                raise IndexError
            # decoder gets encoder output for batch, final hidden output, true labels
            decode_out, decode_hid = decoder(encode_out, encode_hid, y)


            predicted = decode_out.reshape(hyperparameters["batch_size"] * hyperparameters["output_num"], -1)
            # flattens 
            flattened_y = y.reshape(-1)
            loss =criterion(predicted, flattened_y)

            # backwards propagation
            loss.backward()
            e_optimiser.step()
            d_optimiser.step()
            e_optimiser.zero_grad()
            d_optimiser.zero_grad()

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



# hyperparams and params
parameters = {
    "lr": 0.01,
    "n_epochs": 150,
    # measured in epoch numbers
    "plot_every" : 5,
    "print_every" : 10,
    "batch_size": 32,
    "hidden_size": 128,
    "input_size" : 250,
    "output_num": 6,
}

def main():
    # parser = argparse.ArgumentParser()

    # # input size should be = 3

    # args = parser.parse_args()
    split = True
    file = "preprocessed.pt"
    # file = "preprocessed_no_nextx.pt"


    if split:
        dataset = SplitChorales(file)
    else:
        dataset = Chorales(file)

    split_tensors = split_scores(dataset)
    # shuffle = false since data is time contiguous + to learn when an end of piece is
    loader = DataLoader(split_tensors, batch_size=parameters["batch_size"], shuffle=False)

    input_size = parameters["input_size"]
    hidden_size = parameters["hidden_size"]
    output_num = parameters["output_num"]

    encoder = EncoderRNN(input_size, hidden_size)
    decoder = DecoderRNN(hidden_size, input_size, output_num=output_num)

    # loss and optimiser
    e_optimiser = torch.optim.Adam(encoder.parameters(), lr = parameters["lr"])
    d_optimiser = torch.optim.Adam(decoder.parameters(), lr = parameters["lr"])

    criterion = nn.NLLLoss()

    train(encoder, decoder, loader, criterion, e_optimiser, d_optimiser, parameters)




if __name__ == "__main__":
    main()