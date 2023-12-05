
from machine_harmony import main as run




def return_params(lr, n_epochs, dropout, attention, bidirectional, normalisation, batch, hidden):
    return {
    "lr": lr,
    "n_epochs": n_epochs, #maximum number of epochs
    # measured in epoch numbers
    "plot_every" : 2,
    "print_every" : 2,
    "early_stopping": 3, #number of epochs with no improvement after which training is stopped 
    "batch_size": batch,
    "hidden_size": hidden,
    "validation_size": 2, #number of scores in val
    "output_num": 6,
    "SOS_TOKEN": 129, #for the decoder
    "resolution": 8, #used for generation - should be how many items 1 timestep is encoded to
    "iterations": 5, #number of models to run and then average

    # model params
    "dropout": dropout,
    "bidirectional":bidirectional,
    "attention_model": attention, # luong, bahdanau, or None
    "normalisation": normalisation, # dropout, layer (short for layerNorm), or both
    }
# parameters from tuning, used to run, eval and train each individual model 

meta_params = {
    "tokens":"artifacts/tokens.pkl",
    "train_file": "artifacts/preprocessed.pt",
    "test_file":"artifacts/preprocessed_test.pt",
    "type": "both"
}

# model path depends on file name
def complete_run(model_path, attention, bidirectional, normalisation,  n_epochs, batch, hidden, lr, dropout):
    params = return_params(lr, n_epochs, dropout, attention, bidirectional, normalisation, batch, hidden)

    meta_params["model_path"] = model_path

    print("Bidirectional = {}, attention = {}".format(bidirectional, attention))
    run(params, meta_params)


def main():

    complete_run("artifacts/bi-l.pt", "luong", True, "both",  25, 455, 230, 0.01,0.49 )
    complete_run("artifacts/bi-b.pt", "bahdanau", True, "both", 35, 40465, 292, 0.08, 0.67 )
    complete_run("artifacts/bi-n.pt", None, True, "dropout", 35, 4062, 168, 0.06, 0.77 )
    complete_run("artifacts/uni-n.pt", None, False, "dropout", 35, 4035, 974, 0.04, 0.60 )
    complete_run("artifacts/uni-l.pt", "luong", False, "both", 30, 3902, 134, 0.034, 0.30 )
    complete_run("artifacts/uni-b.pt", "bahdanau", False, "both", 65, 4068, 359, 0.04, 0.64 )


main()