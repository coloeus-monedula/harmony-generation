import glob
import pickle
from jsonschema.exceptions import ValidationError, SchemaError
import jsonschema
import json
from matplotlib import pyplot as plt, ticker
import numpy as np
from torch import Tensor
import torch
from os import path, makedirs
import os
import muspy
from music21.stream import Score
from music21 import instrument, converter
from tokeniser import Tokeniser

"""
Converts one or more machine generations (in tensor format) back into a Music21 score, via using MusPy as a intermediary. 
Also allows exporting of audio, and includes functions to plot model graphs and attention weights.
Requires MusPy's music.schema.json file, the tokens data file used to tokenise the dataset, 
and the generated harmony/harmonies in .pt format.
"""

# for use when creating JSON dict, in case using None directly causes issues 
null = None

# have to do this since json.schema.path in muspy's validate func appears to be incorrect
def validate_json(data):
    schema_path = "music.schema.json"
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(data, schema)


# https://www.midi.org/specifications-old/item/gm-level-1-sound-set 
# program midi uses above spec 
def tensor_to_json(tensor: Tensor, folder, filename, token_path,resolution = 8, program_midi = 0):
    data = {}

    metadata = {
        "schema_version": "0.0",
        "title": "Generated Piece",
        "copyright": null,
        "collection":"Generated Dataset",
        "source_filename": filename,
        "source_format": "json"
    }

    track_name = {
        0: "Soprano",
        1:"Alto",
        2:"Tenor",
        3:"Bass",
    }

    tracks = []
    for i in range(4):
        track = add_track(tensor[:,i], track_name.get(i), program_midi)
        tracks.append(track)

    accomp = add_track(tensor[:, 4], "Accompaniment", 0, fb=tensor[:,-1], token_path =token_path , velocity=127)

    tracks.append(accomp)

    data["metadata"] = metadata
    data["resolution"] = resolution
    data["tracks"] = tracks

    try:
        validate_json(data)

        if not path.exists(folder):
            makedirs(folder)

        filepath = path.join(folder, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=6)

    except ValidationError as vE:
        print("Validation error:")
        print(vE)
        raise ValidationError
    except SchemaError as sE:
        print("Schema error:")
        print(sE)
        raise SchemaError
    


# NOTE: due to lack of note-off information, assumes notes are always held and never repeated.
# velocity = volume of part
def add_track(part: Tensor, part_name:str, program_midi:int, velocity = 64, fb:Tensor = None, token_path = None):
    track = {
        "program": program_midi,
        "is_drum": False,
        "name" : part_name 
    }

    # if fb is not None, increment alongside part
    # get reverse lookup alongside 
    if fb is not None:
        if token_path is None:
            raise Exception("Trying to add FBs - however, no tokeniser is given")
        
        tokens = Tokeniser()
        with open(token_path, "rb") as f:
            data = pickle.load(f)
            tokens.load(data)

    notes = []
    converted_fbs = []
    # in timesteps
    # each item, +1 timestep
    time = 0
    current_pitch = part[0].item()
    current_duration = 0
    index = 0
    for pitch in part:

        if (pitch.item() != current_pitch):
            # save previous pitch to list, if not silence 
            if (current_pitch != SILENCE):
                note = {
                    "time": time,
                    "duration": current_duration,
                    "pitch": current_pitch,
                    "velocity": velocity
                }

                notes.append(note)

                if fb is not None:
                    # add a FB for every new note, including None
                    fb_num = fb[time].item()
                    fb_str = tokens.get_with_commas(fb_num)

                    lyric = {
                        "time": time,
                        "lyric": fb_str
                    }
                    converted_fbs.append(lyric)

            # change to new pitch, increment timestep counter
            current_pitch = pitch.item()
            time +=current_duration
            # start at 1 since being in the list means that pitch will at the very least have 1 timestep duration
            current_duration = 1
        else:
            current_duration+=1  
        
        index +=1

    track["notes"] = notes
    if fb is not None and token_path is not None:
        track["lyrics"] = converted_fbs
    return track



def muspy_to_music21(filename,json_folder="generated_JSON", show=False) -> Score:
    filepath = path.join(json_folder,filename+".json")
    muspy_obj = muspy.load_json(filepath)
    m21 = muspy.to_music21(muspy_obj)

    # if notation == "None" or "Unknown", make = None
    # manually add FBs to music21 object
    fbs = muspy_obj.tracks[-1].lyrics
    accomp = m21.parts[-1].notes
  
    for i in range(len(accomp)):
        note = accomp[i]
        fb = fbs[i].lyric

        if fb == "None":
            continue
        else:
            fb_split = fb.split(",")
            for i in range (len(fb_split)):
                single = fb_split[i]
                note.addLyric(single, i+1)

    if (show):
        m21.show()

    return m21


def export_audio(filename, json_folder, sound_folder, from_muspy = True):
    try:
        if from_muspy:
            score = muspy_to_music21(filename,json_folder )
        else:
            score = converter.parse(filename)
            # filename passed in will be full path so just take last part for naming midi
            filename = path.basename(filename)
    except FileNotFoundError:
        print(filename, "not found, audio not exported")
        return

    if not path.exists(sound_folder):
        makedirs(sound_folder)
    filepath = path.join(sound_folder, filename+".midi")

    # convert first to m21 so exported sound can sound the same as the realised FB
    for part in score.parts:
        part.insert(0, instrument.Choir())
    for el in score.parts[-1].recurse():
        if 'Instrument' in el.classes:
            el.activeSite.replace(el, instrument.Contrabass())

    # transpose an octave down - music21 looks at the  notes on the score and ignores the fact the actual pitch is an octave lower, unlike musescore
    # meanwhile, muspy encodes actual pitch presumably
    if from_muspy == False:
        score.parts[-1].transpose("P-8", inPlace=True)
        
    score.write("midi", fp=filepath)


# converts all generated in the temp folder to JSON -> music21 -> midi
# also adds original audio for compariso
# prefix_num determines how many characters to cut off to get file for original audio
# NOTE: by default assumes only u- and b- prefix audios are in folder
def convert_all_generated(folder = "temp", tokens = "artifacts/230_tokens.pkl", og_folder = "./chorales/FB_source/musicXML_master", prefix_num = 2, convert_original = True):
    search = path.join(folder, "*.pt")
    files = glob.glob(search)

    for file in files:
        basename = path.basename(file)
        basename = path.splitext(basename)[0]
        dataset = torch.load(file)
        generated = dataset["generated"]

        tensor_to_json(generated, "generated_JSON", basename+".json", token_path=tokens )
        export_audio(basename, "generated_JSON", "audio")

        # original audio
        if convert_original:
            print("Converting original audio for", basename)
            og_audio_name = basename[prefix_num:]
            og_path = os.path.join(og_folder, og_audio_name)
            export_audio(og_path, "N/A", "audio", from_muspy=False)



def plot(points, plot_epoch, type, title ):
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

    ax.set_title(title)

    plt.show()

# https://github.com/adeveloperdiary/DeepLearning_MiniProjects/blob/master/Neural_Machine_Translation/NMT_RNN_with_Attention_Inference.py
# references above code 
def plot_attention(attention, input, labels, title = "Attention Weights"):
    fig, ax = plt.subplots()

    # get attention into matrix of 3 by 6 instead of 6 by 3
    attention = np.transpose(attention.numpy())

    heatmap = ax.matshow(attention, cmap="bone")
    fig.colorbar(heatmap)

    ax.tick_params(labelsize=10)#
    ax.set_yticklabels([''] + [int(i.item()) for i in input] + [''])
    ax.set_xticklabels([''] + [int(i.item()) for i in labels] + [''])#

    # label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.ylabel('Input Sequence')
    plt.xlabel('Output Sequence')
    plt.title(title)

    plt.show()
    plt.close()

# same value as used in preprocessing and tokenisation
SILENCE = 128

def main():
    # NOTE: converts single muspy obj to music21. needs JSON file generated from tensor_to_json first
    # muspy_to_music21("b-BWV_245.15_FB.musicxml")
    # export_audio(filename, "generated_JSON", "audio")


    # NOTE: code for generating a single audio
    # generated_path = "temp/generated.pt"
    # resolution = 8
    # generated = torch.load(generated_path)
    # filename = "test"
    # tensor_to_json(generated, "generated_JSON", filename+".json", "artifacts/tokens.pkl")

    # export_audio(filename, "generated_JSON", "audio")

    # NOTE: Converts all the files in temp folder to audio
    prefix_num = 0
    convert_original = True
    convert_all_generated(prefix_num= prefix_num, convert_original=convert_original)

if __name__ == "__main__":
    main()

