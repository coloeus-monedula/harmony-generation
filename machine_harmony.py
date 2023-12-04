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
import matplotlib.pyplot as plt, matplotlib.ticker as ticker
from model import *

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

plt.switch_backend('agg')

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
def get_new_model(token_path, bidirectional, attention_model, dropout_p, SOS_token):
    # calculate input size dynamically by the tokeniser
    # add 1 to avoid out of index errors since 0 is also used as a token
    with open(token_path, "rb") as f:
        tokens.load(pickle.load(f))

    input_size = tokens.get_max_token() + 1
    hidden_size = parameters["hidden_size"]
    output_num = parameters["output_num"]

    encoder = EncoderRNN(input_size, hidden_size, bidirectional=bidirectional, dropout_p=dropout_p)
    decoder = DecoderRNN(hidden_size, input_size, output_num=output_num, attention_model=attention_model, dropout_p=dropout_p, device=device, SOS_token=SOS_token)
    encode_decode = EncoderDecoder(encoder, decoder)

    # loss and optimiser
    optimiser = torch.optim.Adam(encode_decode.parameters(), lr = parameters["lr"])
    criterion = nn.CrossEntropyLoss()

    return encode_decode, optimiser, criterion


def generate(model: EncoderDecoder, score: tuple[torch.Tensor, torch.Tensor], hyperparameters, show_attention = False):

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

            output, _, weights = model(x, None)

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

    if show_attention:
        if weights is None:
            print("Can't show attention matrix as there are no attention weights")
        else:
            # TODO
            pass


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


def plot(points, plot_epoch, type):
    plt.figure()
    fig, ax = plt.subplots()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))

    # multiply indexes by plot_epoch + 1 to get epoch numbers 
    x = [(i * plot_epoch) + 1 for i, _ in enumerate(points)]

    ax.plot(x, points)

    ax.set_xlabel("Epoch")
    if type == "loss":
        ax.set_ylabel("Loss")
    elif type == "accuracy":
        ax.set_ylabel("Accuracy")


    plt.show()

def plot_attention(weights, input, output):
    plt.figure()
    fig, ax = plt.subplots()

    heatmap = ax.matshow(weights.cpu().numpy(), cmap="bone")
    fig.colorbar(heatmap)

    ax.set_xticklabels(input, rotation = 90)
    ax.set_yticklabels(output)#

    # label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Input Sequence')
    plt.ylabel('Output Sequence')
    plt.title('Attention Weights')

    plt.show()


# hyperparams and params
parameters = {
    "lr": 0.01,
    "n_epochs": 101, #maximum number of epochs
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
    "dropout": 0.2,
    "bidirectional":True,
    "attention_model": None, # luong, bahdanau, or None
}

tokens = Tokeniser()

def main():
    # parser = argparse.ArgumentParser()

    # # input size should be = 3

    # args = parser.parse_args()
    eval = False

    model_path = "content/bi-none.pt"
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
        
        model, optimiser, _ = get_new_model(token_path, parameters["bidirectional"], parameters["attention_model"], parameters["dropout"], parameters["SOS_TOKEN"])
        checkpoint = torch.load(model_path, map_location=device)
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
            model, optimiser, criterion = get_new_model(token_path, parameters["bidirectional"], parameters["attention_model"], parameters["dropout"], parameters["SOS_TOKEN"])

            model.to(device)

            result = train(model, loader, criterion, optimiser, parameters)
            results.append(result)

        # sort by order of increasing final loss
        results.sort(key = lambda x: x[1] )

        avg_final_loss = 0
        avg_final_accuracy = 0
        avg_length = math.ceil(parameters["n_epochs"]/parameters["plot_every"])

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
        }, model_path)

        print("Average final loss across {} iterations: {}".format(iters, avg_final_loss))
        print("Average final accuracy across {} iterations: {}".format(iters, avg_final_accuracy))







if __name__ == "__main__":
    main()