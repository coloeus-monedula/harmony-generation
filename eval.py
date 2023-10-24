from music21.figuredBass import possibility
from music21 import chord
from music21 import *


# to differentiate close vs open harmony
max_semitone_separation = 12

# chord location costs
chord_costs = {
    # assume to fall under "smooth connection of each part" metric
    "close": -4,
    "doubled_leading": 1,
    "not_in_range": 4 
}

# TODO: thresholding to see what is determined as "smooth connection"?
# parallel fifth is successive fifth, hidden _ is a subcategory of successive _s
transition_costs = {
    "parallel_8th": 5,
    "parallel_8th": 5,
    "parallel_5th": 3,
    "hidden_5th": 3,
    "hidden_8th": 3,


}

# TODO: pitches should be like midi numbers
# type = either list of midi numbers or Pitch objects
# input_type = parts or chords
# if parts - returns a new dict with the same keys, but has the part pitches as values instead 

def get_pitches_music21_parts(parts: dict, format) -> dict:
    pitches_dict = {}
    for key, part in parts.items():
        pitches = part.pitches
        if (format == "obj"):
            pitches_dict[key] = pitches
        elif( format == "number"):
            # convert to midi to match machine learning pianorolls which also tend to use midi vals
            midi_pitches = [pitch.midi for pitch in pitches]
            pitches_dict[key] = midi_pitches

    return pitches_dict


# if chords - returns list of tuples of Pitches 1:1 to Chords
def get_pitches_music21_chords(chords: list, format) -> list:
    pitches_list = []
    for aChord in chords.recurse().getElementsByClass(chord.Chord):
        pitches = aChord.pitches
        if (format == "obj"):
            pitches_list.append(pitches)
        elif (format =="number"):
            midi_pitches = (pitch.midi for pitch in pitches)
            pitches_list.append(midi_pitches)

    return pitches_list

# TODO: rules eval - use the figuredBass.possibility object to check manually
# negative rules increase cost bc less likely
# https://web.mit.edu/music21/doc/moduleReference/moduleChord.html#chord also use the Chord() funcs

# http://web.mit.edu/music21/doc/usersGuide/usersGuide_09_chordify.html chordify

def rules_based_eval(score, chord_checks, trans_checks, local_adjust = 5, trans_adjust = 1):
    harmony_costs = []

    # "If a Score or Part of Measures is provided, a Stream of Measures will be returned"
    chords = score.chordify(addPartIdAsGroup = True)

    # TODO: convert to chordify
    # iterate through, convert to pitch (store in Score format)? use Chord.pitches
    # get big list, keep track of index and total length
    # up to and excluding i = n-1
    # run first index through chord eval, then run both through transition eval to get cost
    # update running score
    pitches = get_pitches_music21_chords(chords, format="obj")
    size = len(pitches)

    # everything up to and excluding the n-1 chord
    for i in range(0, size - 1):
        first = chords[i]
        second = chords[i+1]

        local_cost = eval_chord(first, chord_checks, local_adjust)
        transition_cost = trans_adjust * eval_transitions(first, second, trans_checks)

        total = local_cost+transition_cost
        harmony_costs.append(total)

    
    # n - 1 chord
    first = chords[size - 2]
    second = chords[size - 1]
    local_cost = eval_chord(first, chord_checks, local_adjust) + eval_chord(second, chord_checks, local_adjust)
    transition_cost = eval_transitions(first, second, trans_checks)

    # at the very end sum everything up
    # return dict of  


# chord placement rules
def eval_chord(chord, checks: dict(str, bool), adjust_factor):
    


# transition rules
def eval_transitions(first, second, checks: dict(str, bool)):





# TODO: decide whether the melody line also needs to be evaluated, or just the realised harmonies?
# TODO: add some sort of tag to music21 object so i know which part is which

# TODO: how to compare to og if realised part has extracted notes due to figured bass?
# a note by note comparison???

# def similarity_eval():




def main():
    print("hi")
    #TODO: get stuff from manual_harmony
    # subprocess?

    #  "proper_range": bool,
    # "dbl_leading_note": bool,
    # "close_position": bool,
    # "unprepared_7th": bool,
    # "unprepared_9th": bool


    #     "hidden_8th": bool,
    # "hidden_unison": bool,
    # "parallel_5th": bool,
    # "parallel_8th": bool,
    # "parallel_2nd": bool,

if __name__ == "__main__":
    main()