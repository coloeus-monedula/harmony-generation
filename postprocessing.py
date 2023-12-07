import glob
import pickle
from jsonschema.exceptions import ValidationError, SchemaError
import jsonschema
import json
from torch import Tensor
import torch
from os import path, makedirs
import os
import muspy
from music21.stream import Score
from music21 import instrument, converter

from tokeniser import Tokeniser

null = None

# have to do this since json.schema.path in muspy's validate func appears to be incorrect
def validate_json(data):
    schema_path = "music.schema.json"
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    jsonschema.validate(data, schema)


# https://www.midi.org/specifications-old/item/gm-level-1-sound-set 
# program midi uses above spec 
def tensor_to_json(tensor: Tensor, folder, filename, token_path,resolution = 8, program_midi = 55):
    data = {}

    # TODO: change generated piece to piece the FB is taken from?
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
    # reed organ sound
    accomp = add_track(tensor[:, 4], "Accompaniment", 17, fb=tensor[:,-1], token_path =token_path )

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
    
    # return data

    # TODO: get columns by index number - put into separate rows
    # S, A, T, B, Acc - discard FB
    # new track for each column: program, is_drum, name is set
    # for each track: start timestep counter at 0.
    # have a "current pitch" variable and a "current duration" variable
    # every time pitch stays the same, + 1 to current dur
    # every time pitch variable changes, save the note time pitch  (velocity?) and duration, change current pitch, and reset current duration
    # NOTE: ASSUMES NO REPEATED CONSECUTIVE NOTES
def add_track(part: Tensor, part_name:str, program_midi:int, velocity = 64, fb:Tensor = None, token_path = None):
    track = {
        "program": program_midi,
        "is_drum": False,
        "name" : part_name 
    }


    # TODO: if fb is not None, increment alonside part
    # get reverse lookup alongside 
    if fb is not None:
        if token_path is None:
            raise Exception("Trying to add FBs - however, no tokeniser is given")
        
        tokens = Tokeniser()
        with open(token_path, "rb") as f:
            data = pickle.load(f)
            tokens.load(data)
        reversed = tokens.get_reversed_dict()



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
            # save previous pitch to list, if not silence / pitch 0
            if (current_pitch != 0):
                note = {
                    "time": time,
                    "duration": current_duration,
                    "pitch": current_pitch,
                    "velocity": velocity
                }

                notes.append(note)

            if fb is not None:
                # set as None - ie. don't add to lyrics
                fb_num = fb[index].item()
                fb_str = reversed.get(fb_num, "None")
                if fb_str != "None":
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

def muspy_to_music21(filename, json_folder="generated_JSON", show=False) -> Score:
    filepath = path.join(json_folder,filename+".json")
    muspy_obj = muspy.load_json(filepath)
    # muspy_obj.print()
    m21 = muspy.to_music21(muspy_obj)

    # TODO: MANUALLY ADD BACK FB NOTATIONS HERE
    if (show):
        m21.show()

    return m21

def export_audio(filename, json_folder, sound_folder, from_muspy = True):

    if from_muspy:
        score = muspy_to_music21(filename, json_folder)
    else:
        score = converter.parse(filename)
        # filename passed in will be full path so just take last part for naming midi
        filename = path.basename(filename)


    if not path.exists(sound_folder):
        makedirs(sound_folder)
    filepath = path.join(sound_folder, filename+".midi")

    # convert first to m21 so exported sound can sound the same as the realised FB
    for part in score.parts:
        part.insert(0, instrument.Choir())
    for el in score.parts[-1].recurse():
        if 'Instrument' in el.classes:
            el.activeSite.replace(el, instrument.Piano())

    score.write("midi", fp=filepath)


# converts all generated to JSON -> music21 -> midi
# also adds original audio for comparison
# returns audio
def convert_all_generated(folder = "temp", tokens = "artifacts/230_tokens.pkl", og_folder = "./chorales/FB_source/musicXML_master"):
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
        # NOTE: assumes only u- and b- prefix audios are in folder
        og_audio_name = basename[2:]
        og_path = os.path.join(og_folder, og_audio_name)
        export_audio(og_path, "N/A", "audio", from_muspy=False)


def main():
    # dataset: dict[str, Tensor] = torch.load("preprocessed.pt")
    # items = list(dataset.items())

    # # remove extension
    # filename = path.splitext(items[0][0])[0]

    # tensor_to_json(items[0][1], "generated_JSON", filename+".json")
    # test using preprocessed stuff

    generated_path = "temp/generated.pt"
    resolution = 8
    # generated = torch.load(generated_path)

    # filename = "test"
    # tensor_to_json(generated, "generated_JSON", filename+".json", "artifacts/tokens.pkl")

    # export_audio(filename, "generated_JSON", "audio")

    convert_all_generated()
if __name__ == "__main__":
    main()

