
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


# model path depends on file name
def complete_run(model_path, attention, bidirectional, normalisation,  n_epochs, batch, hidden, lr, dropout, meta_params):
    params = return_params(lr, n_epochs, dropout, attention, bidirectional, normalisation, batch, hidden)

    meta_params["model_path"] = model_path

    print("Bidirectional = {}, attention = {}".format(bidirectional, attention))
    run(params, meta_params)


def main(meta_params, run_type):
    meta_params["type"] = run_type

    if meta_params["token_count"] == 269:
        print("\nToken count max 269 (ie. full tokens).")
        # full tokens
        complete_run("artifacts/bi-l.pt", "luong", True, "both",  25, 455, 230, 0.01,0.49,meta_params)
        complete_run("artifacts/bi-b.pt", "bahdanau", True, "both", 35, 4065, 292, 0.08, 0.67,meta_params)
        complete_run("artifacts/bi-n.pt", None, True, "dropout", 35, 4062, 168, 0.06, 0.77,meta_params)
        complete_run("artifacts/uni-n.pt", None, False, "dropout", 35, 4035, 974, 0.04, 0.60,meta_params)
        complete_run("artifacts/uni-l.pt", "luong", False, "both", 30, 3902, 134, 0.034, 0.30,meta_params)
        complete_run("artifacts/uni-b.pt", "bahdanau", False, "both", 65, 4068, 359, 0.04, 0.64,meta_params)
    elif meta_params["token_count"] == 250:
        print("\nToken count 250.")
        # 250 tokens
        complete_run("artifacts/bi-l-250.pt", "luong", True, "dropout",  60, 3567, 136, 0.03,0.36,meta_params)
        complete_run("artifacts/bi-b-250.pt", "bahdanau", True, "layer", 30, 3758, 348, 0.02, 0.30,meta_params)
        complete_run("artifacts/bi-n-250.pt", None, True, "dropout", 40, 3819, 348, 0.06, 0.65,meta_params)
        complete_run("artifacts/uni-n-250.pt", None, False, "dropout", 70, 4035, 141, 0.06, 0.61,meta_params)
        complete_run("artifacts/uni-l-250.pt", "luong", False, "dropout", 25, 716, 274, 0.03, 0.50,meta_params)
        complete_run("artifacts/uni-b-250.pt", "bahdanau", False, "layer", 80, 3698, 294, 0.012, 0.36,meta_params)

    elif meta_params["token_count"] == 230:
        print("\nToken count 230.")
        complete_run("artifacts/bi-l-230.pt", "luong", True, "both",  15, 3736, 170, 0.013,0.6,meta_params )
        complete_run("artifacts/bi-b-230.pt", "bahdanau", True, "dropout", 15, 3722, 173, 0.07, 0.66,meta_params)
        complete_run("artifacts/bi-n-230.pt", None, True, "both", 80, 3660, 283, 0.03, 0.33,meta_params)
        complete_run("artifacts/uni-n-230.pt", None, False, "layer", 95, 3147, 320, 0.013, 0.72,meta_params)
        complete_run("artifacts/uni-l-230.pt", "luong", False, "dropout", 75, 3768, 167, 0.011, 0.66,meta_params)
        complete_run("artifacts/uni-b-230.pt", "bahdanau", False, "both", 100, 3881, 150, 0.064, 0.30,meta_params)



# parameters from tuning, used to run, eval and train each individual model 


meta_params_full = {
    "token_count": 269,
    "tokens":"artifacts/tokens.pkl",
    "train_file": "artifacts/preprocessed.pt",
    "test_file":"artifacts/preprocessed_test.pt",
    "prefix":""
}

meta_params_250 = {
    "token_count": 250,
    "tokens":"artifacts/250_tokens.pkl",
    "train_file": "artifacts/250_preprocessed.pt",
    "test_file":"artifacts/250_preprocessed_test.pt",
    "prefix":""
}

meta_params_230 = {
    "token_count": 230,
    "tokens":"artifacts/230_tokens.pkl",
    "train_file": "artifacts/230_preprocessed.pt",
    "test_file":"artifacts/230_preprocessed_test.pt",
    "prefix":""
}


main(meta_params_full, run_type="train")
main(meta_params_250, run_type="train")
main(meta_params_230, run_type="train")
main(meta_params_full, run_type="eval")
main(meta_params_250, run_type="eval")
main(meta_params_230, run_type="eval")