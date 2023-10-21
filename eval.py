from music21.figuredBass import possibility

chord_costs = {

}

# TODO: thresholding to see what is determined as "smooth connection"?
transition_costs = {

}

# returns a new dict with the same keys, but has the part pitches as values instead 
# TODO: pitches should be like midi numbers
# type = either midi numbers or string
def get_pitches_music21(parts: dict, format):
    pitches = {}

    for key, part in parts.items():
        pitch = part.pitches
        if (format == "string"):
            pitches[key] = pitch
        elif( format == "number"):
            # convert to midi to match machine learning pianorolls which also tend to use midi vals
            midi_pitch = [note.midi for note in pitch]
            pitches[key] = midi_pitch
    
    return pitches

# TODO: rules eval - use the figuredBass.possibility object to check manually
# negative rules increase cost bc less likely
def rules_based_eval():

# chord placement rules
def eval_chord():

# transition rules
def eval_transitions():


# TODO: decide whether the melody line also needs to be evaluated, or just the realised harmonies?
# TODO: add some sort of tag to music21 object so i know which part is which

# TODO: how to compare to og if realised part has extrac notes due to figured bass?
# a note by note comparison???

# def similarity_eval():




def main():
    #TODO: get stuff from manual_harmony
    # subprocess?

if __name__ == "__main__":
    main()