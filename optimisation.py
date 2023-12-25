# hyperparameter tuning

import argparse
import joblib
import torch
import optuna
from optuna.trial import TrialState
from functools import partial
from machine_harmony import get_new_model, split_train_val
from model import *
from local_datasets import PytorchSplitChoralesDataset as SplitChorales
from torch.utils.data import DataLoader

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")




# aim is to maximise test accuracy
# skeleton based of this tutorial
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py 
def objective(trial: optuna.Trial, params):
    # add params to tune
    params["dropout"] = trial.suggest_float("dropout_p", 0.3, 0.8)
    params["lr"] = trial.suggest_float("lr", 0.01, 0.1)
    params["hidden_size"] = trial.suggest_int("hidden_size",128,1024,log=True)
    params["batch_size"] = trial.suggest_int("batch_size", 256, 4096, log=True )
    params["normalisation"] = trial.suggest_categorical("normalisation", ["both","layer","dropout"] )
    n_epochs = trial.suggest_int("epochs", 10, 100, step=5)


    train_loader, val_loader = get_loaders(params["batch_size"])
    model, optimiser, criterion = get_new_model(token_path, params)

    model.to(device)
    # train loop
    # rewrite here so pruning can be done via optuna and to cut out unnecessary bits
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            output, hidden, _ =  model(x, y)

            # [batch size, num outputs, output size/classes] -> [batch size x num outputs, classes]
            predicted = output.reshape(params["batch_size"] * params["output_num"], -1)
            # flattens 
            flattened_y = y.reshape(-1)
            loss =criterion(predicted, flattened_y)

            # backwards propagation
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                output, _, _ =  model(x, y)
                # [batch size, num outputs, output size/classes] -> [batch size x num outputs, classes]
                flattened_y = y.reshape(-1)

                preds_labels = torch.argmax(output, -1)
                preds_labels = preds_labels.reshape(params["batch_size"] * params["output_num"])

                correct += (preds_labels == flattened_y).sum().cpu().item()
                total += preds_labels.size(0)

        val_accuracy = 100* correct/total
        trial.report(val_accuracy, epoch)


        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return val_accuracy


def get_loaders(batch_size):
    dataset = SplitChorales(train_file)
    train_dataset, val_dataset = split_train_val(dataset, validation_size, batch_size)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_loader, val_loader




def run_trial(bidirectional, attention, n_trials, save):
    params = {
        "bidirectional": bidirectional,
        "attention_model": attention,
        "SOS_TOKEN": SOS_token,
        "token_path": token_path,
        "output_num": output_num,
        
    }

    objective_partial = partial(objective, params = params)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective_partial, n_trials=n_trials)

    joblib.dump(study, save)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    best_params = study.best_params
    print("Best Hyperparameters for bidirectional = {}, attention = {}: ".format(bidirectional, attention), best_params)



SOS_token = 129
plot_every = 2
print_every = 2
early_stopping = 5
output_num = 6
validation_size = 2
# n_epochs = 30
early_stopping = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter tuning")
    parser.add_argument("tokens", help = "How many tokens used to preprocess the scores")

    args = parser.parse_args()
    num = args.tokens

    token_path = "artifacts/{}_tokens.pkl".format(num)
    train_file = "artifacts/{}_preprocessed.pt".format(num)
    print("Token number = {}".format(num))
    # run_trial(True, "luong", 70, "artifacts/bi-l-{}.joblib".format(num))
    # run_trial(True, "bahdanau", 70, "artifacts/bi-b-{}.joblib".format(num))
    # run_trial(True, None, 70, "artifacts/bi-None-{}.joblib".format(num))
    # run_trial(False, None, 70, "artifacts/uni-None-{}.joblib".format(num))
    run_trial(False, "luong", 70, "artifacts/uni-l-{}.joblib".format(num))
    run_trial(False, "bahdanau", 70, "artifacts/uni-b-{}.joblib".format(num))



