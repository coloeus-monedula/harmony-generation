import argparse
from manual_harmony import convert_music21
import torch.nn as nn
import torch
from local_datasets import PytorchChoralesDataset as Chorales, PytorchSplitChoralesDataset as SplitChorales
from torch.utils.data import DataLoader, TensorDataset
import time, math
from tokeniser import Tokeniser
import dill as pickle
from sklearn.metrics import accuracy_score
import numpy as np

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


# encoder decoder structure follows https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html and supervisor code
# NOTE: allows for n_layers > 1 but doesn't necessary work with it
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,  bidirectional, n_layers = 1, dropout_p = 0.05,) -> None:
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        # turn input into a hidden_size sized vector, use to try and learn relationship between pitches and FB notations 
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size,num_layers= n_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.embedding(input)
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

            return output, (torch.cat(hidden_layers,0), torch.cat(cell_layers, 0))

        else:

            return output, hidden
    
    # check embedding
    def get_embedding(self, x):
        return self.embedding(x)

# Attention module based off
# https://github.com/rawmarshmellows/pytorch-batch-luong-attention/blob/master/models/luong_attention/luong_attention.py
class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, decoder_output, encoder_outputs):
        # for us, decoder output = batch size, 1, hidden size
        # encoder_output = batch size, max input (default 3), hidden size
        # change to = batch size, hidden, length

        attn_energies = torch.bmm(self.attn(decoder_output), encoder_outputs.permute(0, 2, 1))


        # Batch size, 1, max input length
        return nn.functional.softmax(attn_energies, dim = -1)

    def score(self, hidden, encoder_output):

        if self.method == 'general':
            energy = self.attn(encoder_output).view(-1)
            energy = hidden.view(-1).dot(energy)
            return energy
    

class DecoderRNN(nn.Module):

    # output size typically equals encoder's input size
    # NOTE: "timesteps" are columns of output.
    def __init__(self,  hidden_size, output_size, attention_model, output_num = 6,  n_layers =1, dropout_p = 0.1) -> None:
        super(DecoderRNN, self).__init__()


        self.output_num = output_num
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        # self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

        if attention_model is not None:
            self.attention = Attention(attention_model, hidden_size)
            self.concat = nn.Linear(2*hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size*n_layers, output_size)
        

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        batch_size = encoder_outputs.size(0)

        # second dimension = 1 - means we are processing 1 timestep at a time
        decoder_input = torch.empty(batch_size, self.n_layers, dtype=torch.long, device=device).fill_(parameters["SOS_TOKEN"])

        # context vector from encoder used as initial hidden state
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        # process num_outputs columns at a time - typically, the 6 values we want
        # for the whole of the batch
        for i in range(self.output_num):
            decoder_output, decoder_hidden, attention_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # feed target as next input/column 
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # use own prediction as next input 
                # use whilst generating/predicting - we don't want model to know true values
                vals, indices = decoder_output.topk(1)
                decoder_input = indices.squeeze(-1).detach()


        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs, decoder_hidden, attention_weights
    
    # for a single input - embed, rnn, use linear output layer
    def forward_step(self, input, hidden, encoder_outputs):
        output = self.embedding(input)
        if hasattr(self, "attention"):
            output = self.dropout(output)


        output, hidden = self.rnn(output, hidden)
        if hasattr(self, "attention"):
            # attention weight calculations from current rnn output
            # print(output.shape, encoder_outputs.shape)
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

def time_since(since):
    now = time.time()
    secs = now-since
    mins = math.floor(secs/60)
    secs -= mins * 60
    return '%dmins %dsecs' % (mins, secs)

def train(model:EncoderDecoder, loader:DataLoader, criterion:nn.CrossEntropyLoss, optimiser:torch.optim.Adam, hyperparameters):

    all_losses = []
    all_accuracies = []
    start = time.time()

    for epoch in range(hyperparameters["n_epochs"]):
        total_loss = 0
        # if batches = 1 then num_batches = number of scores
        # num_batches = len(loader)
        # number of rounds
        num_rounds = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            output, hidden, _ =  model(x, y)

            # [batch size, num outputs, output size/classes] -> [batch size x num outputs, classes]
            predicted = output.reshape(hyperparameters["batch_size"] * hyperparameters["output_num"], -1)
            # flattens 
            flattened_y = y.reshape(-1)
            loss =criterion(predicted, flattened_y)

            preds_labels = torch.argmax(output, -1)
            preds_labels = preds_labels.reshape(hyperparameters["batch_size"] * hyperparameters["output_num"])

            # backwards propagation
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            total_loss += loss.item()
            num_rounds+=1
            accuracy = accuracy_score(flattened_y.cpu().numpy(), preds_labels.cpu().numpy())

        if epoch % hyperparameters["print_every"] == 0:
            print('Epoch: {}/{},'.format(epoch+1, hyperparameters["n_epochs"]), end=" ")
            print(time_since(start),end = "......................... ")
            print("Loss: {:4f}".format(total_loss/num_rounds), end=", ")
            print("Accuracy: {:4f}".format(accuracy))

        # add average loss per epoch across all batches to plot graph
        if epoch % hyperparameters["plot_every"] == 0:
            all_losses.append(total_loss/num_rounds)
            all_accuracies.append(accuracy)

    # average total loss at the end 
    final_loss = total_loss/num_rounds

    # return to cpu
    model = model.cpu()

    return model, final_loss, all_losses, all_accuracies


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
def pad(split_tensors, batch_size):
    modulo = len(split_tensors) % batch_size
    to_pad = batch_size - modulo

    total_padding_x, total_padding_y = pad_x(to_pad), pad_y(to_pad)
    
    total_padding = TensorDataset(total_padding_x, total_padding_y)
    split_tensors = split_tensors + total_padding
    return split_tensors

def pad_x(to_pad, tensor = [0,0,0]):
    total_padding_x = []
    for i in range(to_pad):
        total_padding_x.append(torch.tensor(tensor).long())

    total_padding_x = torch.stack(total_padding_x).long()

    return total_padding_x

def pad_y(to_pad, tensor = [0,0,0,0,0,0]):
    total_padding_y = []
    for i in range(to_pad):
        total_padding_y.append(torch.tensor(tensor).long())

    total_padding_y = torch.stack(total_padding_y).long()

    return total_padding_y

# returns clean model and optimiser
def get_new_model(token_path, bidirectional, attention_model):
    # calculate input size dynamically by the tokeniser
    # add 1 since 0 is also used as a token to avoid out of index errors
    with open(token_path, "rb") as f:
        tokens.load(pickle.load(f))

    input_size = tokens.get_max_token() + 1
    hidden_size = parameters["hidden_size"]
    output_num = parameters["output_num"]

    encoder = EncoderRNN(input_size, hidden_size, bidirectional=bidirectional)
    decoder = DecoderRNN(hidden_size, input_size, output_num=output_num, attention_model=attention_model)
    encode_decode = EncoderDecoder(encoder, decoder)

    # loss and optimiser
    optimiser = torch.optim.Adam(encode_decode.parameters(), lr = parameters["lr"])
    criterion = nn.CrossEntropyLoss()

    return encode_decode, optimiser, criterion


def generate(model: EncoderDecoder, score: tuple[torch.Tensor, torch.Tensor], hyperparameters):

    model.eval()
    model.to(device)
    
    batch_size = hyperparameters["resolution"]
    output_size = hyperparameters["output_num"]

    dataset = TensorDataset(score[0], score[1])
    dataset = pad(dataset, batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle = False)

    # TODO: if we want some randomness as to whether model uses true S,Acc,FB+1 vals or predicted S,Acc,FB+1 vals implement this here
    generated_ATB = []
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            output, _ = model(x, None)

            # A,T,B, S+1, Acc+1, FB+1
            preds = torch.argmax(output, -1)

            # use SAccFb to predict next steps if needed
            ATB, SAccFb = torch.tensor_split(preds, [int(output_size/2)], 1 )

            generated_ATB.extend(ATB)

            # for analysis against y for accuracy 
            # TODO: analyse everything? or just ATB?
            preds = preds.reshape(batch_size*output_size)
            flattened_y = y.reshape(-1)

            # move to cpu
            correct += (preds == flattened_y).sum().cpu().item()
            total += preds.size(0)

    accuracy = 100 * correct / total
    print("Accuracy on test chorale: {:4f}".format(accuracy))

    # convert to torch
    generated_ATB = torch.stack(generated_ATB).long()

    return accuracy, join_score(score[0], generated_ATB)

def join_score(x: torch.Tensor, y: torch.Tensor):

    # adds padding to end of x if needed
    if (len(x) != len(y)):
        diff = len(y) - len(x)
        padding = pad_x(diff)
        x = torch.cat((x, padding))

    x = x.to(device)
    y = y.to(device)
    # NOTE: assumes S, Acc, FB order in x, and A, T, B order in y 
    s = x[:,0]
    acc = x[:,1]
    fb = x[:,2]

    a = y[:,0]
    t = y[:,1]
    b = y[:,2]

    generated = torch.stack([s,a,t,b,acc,fb], dim = 1)
    # add a line of silence
    silence = torch.tensor([[0,0,0,0,0,0]]).to(device)
    generated = torch.cat((generated, silence)) 

    return generated


# hyperparams and params
parameters = {
    "lr": 0.01,
    # "n_epochs": 100,
    "n_epochs": 201, #maximum number of epochs
    # measured in epoch numbers
    "plot_every" : 5,
    "print_every" : 10,
    "batch_size": 128,
    "hidden_size": 256,
    #the unknown token is set as 250 and if you set input size = unknown token num it gives an out of index error when reached
    # # possibly because 0 is also used as a token so off by 1
    # "input_size" : 252, 
    "output_num": 6,
    "SOS_TOKEN": 129, #for the decoder
    "resolution": 8, #used for generation - should be how many items 1 timestep is encoded to
    "iterations": 5, #number of models to run and then average
    "dropout": 0.1,
    "bidirectional":True,
    "attention_model": 'concat',
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
    train_file = "content/preprocessed.pt"
    test_file = "content/preprocessed_test.pt"

    # TODO: make this the chorale name
    generated_path = "temp/generated.pt"


    if eval:
        print("Evaluating model.")
        if split:
            test_dataset = SplitChorales(test_file)
        else:
            # NOTE: following model code assumes SplitChorales and doesn't account for Chorales
            test_dataset = Chorales(test_file)
        
        model, optimiser, _ = get_new_model(token_path, parameters["bidirectional"], parameters["attention_model"])
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        # optimiser.load_state_dict(checkpoint["optimiser"])


        for i in range(len(test_dataset)):
            accuracy, generated = generate(model, test_dataset[i], parameters)

        # save last one for now
        generated = generated.cpu()
        torch.save(generated, generated_path)

        print(generated)

        

    else:
        print("Training model.")
        if split:
            dataset = SplitChorales(train_file)
        else:
            # NOTE: following model code assumes SplitChorales and doesn't account for Chorales
            dataset = Chorales(train_file)
        
        split_tensors = split_scores(dataset)
        split_tensors = pad(split_tensors, parameters["batch_size"])

        # shuffle = false since data is time contiguous + to learn when an end of piece is
        loader = DataLoader(split_tensors, batch_size=parameters["batch_size"], shuffle=False)

        results = []
        iters = parameters["iterations"]

        for i in range(iters):
            print("Iteration {}.".format(i+1))
            model, optimiser, criterion = get_new_model(token_path, parameters["bidirectional"], parameters["attention_model"])
            model.to(device)

            result = train(model, loader, criterion, optimiser, parameters)
            results.append(result)

        # sort by order of increasing final loss
        results.sort(key = lambda x: x[1] )

        avg_final_loss = 0
        avg_final_accuracy = 0
        avg_length = int(parameters["n_epochs"]/parameters["plot_every"])

        avg_losses = np.zeros(avg_length, dtype=np.float32)
        avg_accuracies = np.zeros(avg_length, dtype=np.float32)


        for i in range(len(results)):
            model, final_loss, losses, accuracies = results[i]
            avg_final_loss += final_loss
            avg_final_accuracy += accuracies[-1]

            avg_losses += losses
            avg_accuracies += accuracies

        avg_final_loss /= iters
        avg_final_accuracy /=iters
        avg_losses /= iters
        avg_accuracies /= iters

        # save model with lowest loss
        # as well as average running losses and accuracies
        torch.save({
            "model": results[0][0].state_dict(),
            # "optimiser": optimiser.state_dict(),
            "losses": avg_losses,
            "accuracies": avg_accuracies
        }, path)

        print("Average final loss across {} iterations: {}".format(iters, avg_final_loss))
        print("Average final accuracy across {} iterations: {}".format(iters, avg_final_accuracy))







if __name__ == "__main__":
    main()