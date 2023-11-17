from jsonschema.exceptions import ValidationError, SchemaError
import jsonschema
import json
from torch import Tensor
import torch
from os import path, makedirs
import os
import muspy
from music21.stream import Score

null = None

# have to do this since json.schema.path in muspy's validate func appears to be incorrect
def validate_json(data):
    schema_path = "music.schema.json"
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    jsonschema.validate(data, schema)


# https://www.midi.org/specifications-old/item/gm-level-1-sound-set 
# program midi uses above spec 
def tensor_to_json(tensor: Tensor, folder, filename, resolution = 8, program_midi = 53,):
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
    # church organ sound
    accomp = add_track(tensor[:, 4], "Accompaniment", 20)
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
def add_track(part: Tensor, part_name:str, program_midi:int, velocity = 64):
    track = {
        "program": program_midi,
        "is_drum": False,
        "name" : part_name 
    }

    notes = []
    # in timesteps
    # each item, +1 timestep
    time = 0

    current_pitch = part[0].item()
    current_duration = 0
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

            # change to new pitch, increment timestep counter
            current_pitch = pitch.item()
            time +=current_duration
            # start at 1 since being in the list means that pitch will at the very least have 1 timestep duration
            current_duration = 1
        else:
            current_duration+=1  
                  

    track["notes"] = notes
    return track

def muspy_to_music21(filename, json_folder, show=False) -> Score:
    muspy_obj = muspy.load_json(os.path(json_folder,filename+".json") )
    m21 = muspy.to_music21(muspy_obj)

    if (show):
        m21.show()

    return m21

def export_audio(filename, json_folder, sound_folder, extension = ".oga"):
    muspy_obj = muspy.load_json(path.join(json_folder,filename+".json"))

    if not path.exists(sound_folder):
        makedirs(sound_folder)

    # filepath = path.join(sound_folder, filename+extension)
    # muspy.write_audio(filepath, muspy_obj)

    filepath = path.join(sound_folder, filename+".midi")
    muspy.write_midi(filepath, muspy_obj)
    # have to convert to music21 first because write_audio() is buggy
    # m21 = muspy_to_music21(filename, json_folder)

    


# TODO: if we're using data from scores, add info for time sig etc. from that? pass as an object param 
def main():
    dataset: dict[str, Tensor] = torch.load("preprocessed.pt")
    items = list(dataset.items())

    # remove extension
    filename = path.splitext(items[0][0])[0]

    # tensor_to_json(items[0][1], "generated_JSON", filename+".json")
    # test using preprocessed stuff

    export_audio(filename, "generated_JSON", "audio")
if __name__ == "__main__":
    main()

