import argparse
import muspy
from manual_harmony import convert_music21
import torch.nn as nn
import torch
from local_datasets import PytorchChoralesDataset as Chorales, PytorchSplitChoralesDataset as SplitChorales
from torch.utils.data import DataLoader
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



# follows https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, nonlinearity="tanh") -> None:
        super(RNN, self).__init__()

        # dimensions for hidden layers
        self.hidden_size = hidden_size
        # number of RNN layers
        self.n_layers = n_layers

        # will be made up of n_layers of RNN
        # DataLoader does batch number first so switch to batch first
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, nonlinearity=nonlinearity, batch_first=True)
        # reshape to output 
        self.linear = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        batch_size = x.size(0)
        h_n = self.init_hidden(batch_size)

        # final hidden state for each element in batch
        output, h_n = self.rnn(x, h_n)

        # make output stored in same block of memory
        # fit to hidden_size dimensiions to pass through linear layer
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.linear(output)

        # add extra (batch) dimension to output
        output = output[None,:,:]

        return output, h_n

    def init_hidden(self, batch_size):
        # initialise the hidden state - makes a layers x batch size x hidden size sized tensor 
        torch.zeros(self.n_layers, batch_size, self.hidden_size)



class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers) -> None:
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
        output = output[None,:,:]

        return output, h_n


    def init_hidden(self, batch_size):
        # initialise the hidden state - makes a layers x batch size x hidden size sized tensor 
        torch.zeros(self.n_layers, batch_size, self.hidden_size)


def generate_sequences(dataset:SplitChorales|Chorales, split:bool):
    pass


def time_since(since):
    now = time.time()
    secs = now-since
    mins = math.floor(secs/60)
    secs -= mins * 60
    return '%dmins %dsecs' % (mins, secs)

def train(model, loader:DataLoader, criterion:nn.CrossEntropyLoss, optimiser:torch.optim.Adam, hyperparameters):

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

            optimiser.zero_grad()
            output, hidden = model(x)

            # print(output)
            # print(y)

            loss = criterion(output, y)
            loss.backward()
            optimiser.step()

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



def test():
    pass


# hyperparams and params
parameters = {
    "lr": 0.01,
    "n_epochs": 100,
    # measured in epoch numbers
    "plot_every" : 5,
    "print_every" : 10,
    "batch_size": 1, #if data not padded, this is the only one that can be done
    "hidden_size": 10,
    "input_size" : 3,
    "output_size": 6,
    "n_layers" :1,
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

    loader = DataLoader(dataset, batch_size=parameters["batch_size"])

    input_size = parameters["input_size"]
    output_size = parameters["output_size"]
    n_layers = parameters["n_layers"]

    model = LSTM(input_size, output_size, parameters["hidden_size"], n_layers)
    model.to(device)

    # loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = parameters["lr"])

    train(model, loader, criterion, optimiser, parameters)




if __name__ == "__main__":
    main()