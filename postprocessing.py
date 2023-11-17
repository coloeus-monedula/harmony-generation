from muspy.schemas.utils import validate_json
from jsonschema.exceptions import ValidationError
import json
from torch import Tensor
import torch
from os import path, makedirs
import os

null = None

# https://www.midi.org/specifications-old/item/gm-level-1-sound-set 
# program midi uses above spec 
def tensor_to_json(tensor: Tensor, folder, filename, resolution = 8, program_midi = 53,):
    data = {}

    # TODO: change generated piece to piece the FB is taken from?
    metadata = {
        "schema_version": 0.0,
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


    if not path.exists(folder):
        makedirs(folder)

    filepath = path.join(folder, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=6)

    
    try:
        validate_json(filepath)
    except(ValidationError):
        os.remove(filepath)
        raise ValidationError("Invalid JSON - file deleted")
    
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
            # save previous pitch to list
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
            pass
        else:
            current_duration+=1  
                  

    track["notes"] = notes
    return track

# TODO: if we're using data from scores, add info for time sig etc. from that? pass as an object param 
def main():
    dataset: dict[str, Tensor] = torch.load("preprocessed.pt")
    items = list(dataset.items())
    tensor_to_json(items[0][1], "generated_JSON", items[0][0])
    # test using preprocessed stuff
    pass

if __name__ == "__main__":
    main()

