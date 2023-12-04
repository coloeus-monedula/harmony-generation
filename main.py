# main script that does everything:
# given an unknown test file and saved pretrained model:
# generate realisation via fb
# generate prediction via model
# run eval metrics on both and/or save as midi

import torch
from machine_harmony import plot

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")



state = torch.load("content/bi-none.pt", map_location=device)
losses = state["losses"]
accuracies = state["accuracies"]

plot_every = 5

plot(losses, plot_every, "loss")
plot(accuracies, plot_every, "accuracy")