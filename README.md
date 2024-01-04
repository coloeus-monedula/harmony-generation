Final year dissertation project for a Masters in Computer Science at the University of St Andrews.

# Installation
`setup.sh` contains a script to install the project on a university system for Linux. Installation for other systems follows something similar. Below are the instructions for a Linux system, that when used in terminal creates a virtual environment called "venv" and installs everything within there. Note that Python 3.10 is required for certain libraries to function as intended.

```
python -m venv venv
source venv/bin/activate
pip install pipenv
pipenv install
```

# Running
To allow Music21 to find the MuseScore program to open scores in when show() is called, the environment has to be manually set for Linux systems. `env.py` contains code that can be used as an example for how to do this.

Most scripts can be run standalone by calling `python [script.py]` in the terminal. Some require command line arguments; those that do have argparse implemented.
* `manual_harmony.py` and `eval.py` can be run from command line without modifying the script.
* `machine_harmony.py` and `optimisation.py` have command line arguments, but also have parameters close to the bottom of the script that can be modified if required. This is the "parameters" object in `machine_harmony.py`, but `optimisation.py` just has them loose.
* `extract_baseline.py`, `preprocessing.py`, `postprocessing.py`, `eval_testset.py`, and `train_all_variations.py` can be modified by editing the scripts. These all either contain a main() function or use `if __name__ == "__main__":`terminology near the bottom of the script, where relevant parameters to edit will also be. Some scripts, e.g. `postprocessing.py`, will contain commented out lines of code meant to serve as examples of how to use the script.


# Additional Notes
The best performing BiLSTM and LSTM models, and relevant preprocessing files have been included in the `artifacts` folder. The preprocessed dataset and token file for counting the figured bass notations in the filtered dataset have also been included. They come in dictionary format, due to being saved with extra information. Model files have hyperparameter, loss and accuracy data, the token files are meant to be loaded back into a Tokeniser object, and preprocessed files use the name of the music score as a key. Additionally, the test set's MIDI and audio files can be found in the `audio` folder. 

A note on naming conventions: the hyperparameter that controls what regularisation layers are used is called "normalisation" instead. This is technically inaccurate so is noted here for clarity.
