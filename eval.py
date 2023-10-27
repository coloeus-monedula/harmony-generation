from music21.figuredBass import possibility
from music21 import chord
from music21 import *
from nltk import ngrams, FreqDist
from nltk.metrics.distance import jaro_similarity
import pickle


# to differentiate close vs open harmony
max_semitone_separation = 12


# uses https://musictheory.pugetsound.edu/mt21c/VoiceRanges.html as reference and is Bach chorales-specific
# vocal_ranges = {
#     "s": ("D4", "F#5"),
#     "a": ("G3", "C#5"),
#     "t": ("E-3", "F#4"),
#     "b": ("E2", "C4")
# }
vocal_ranges = {
    "s": (293, 740),
    "a": (195, 555),
    "t": (155, 370),
    "b": (82, 262)
}

# chord location costs
chord_costs = {
    # assume to fall under "smooth connection of each part" metric
    "close": -4,
    "doubled_leading": 1,
    # checks for each voice, up to 4 per chord in total
    "not_in_range": 1 
}

# TODO: thresholding to see what is determined as "smooth connection"?
# parallel fifth is successive fifth, hidden _ is a subcategory of successive _s hence the 3 weighting
transition_costs = {
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
    total_costs = 0
    local_list = []
    trans_list = []
    local_total = 0
    trans_total = 0

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

        local_total+=local_cost
        trans_total += transition_cost

        total = local_cost+transition_cost
        total_costs += total
        local_list.append(local_cost)
        trans_list.append(transition_cost)

    
    # n - 1 chord
    first = chords[size - 2]
    second = chords[size - 1]
    local_cost = eval_chord(first, chord_checks, local_adjust) + eval_chord(second, chord_checks, local_adjust)
    transition_cost = eval_transitions(first, second, trans_checks)
    total = local_cost + transition_cost

    local_total+=local_cost
    trans_total+=transition_cost

    total_costs+= total
    local_list.append(local_cost)
    trans_list.append(transition_cost)

    return {
        "total": total_costs,
        "local": local_list,
        "transition": trans_list 
    }


# chord placement rules
# we are ignoring the figure interpretation metrics for now as it's a lot of work
def eval_chord(chord, checks: dict(str, bool), adjust_factor):
    cost = 0
    if (checks["close"] and possibility.upperPartsWithinLimit(chord, max_semitone_separation)):
        cost += chord_costs["close"]
    
    if (checks["range"]):
        (s, a, t, b) = chord
        s = s.frequency
        a = a.frequency
        t = t.frequency
        b = b.frequency

        # check to see if notes fall out of vocal ranges
        if ( s < vocal_ranges["s"][0] or s > vocal_ranges["s"][1]):
            cost +=chord_costs["not_in_range"]
        if ( a < vocal_ranges["a"][0] or s > vocal_ranges["a"][1]):
            cost +=chord_costs["not_in_range"]
        if ( t < vocal_ranges["t"][0] or s > vocal_ranges["t"][1]):
            cost +=chord_costs["not_in_range"]
        if ( b < vocal_ranges["b"][0] or s > vocal_ranges["b"][1]):
            cost +=chord_costs["not_in_range"]


    # TODO: do check if doubled leading note ? - do they mean the 7th or just the semitoneness? 
    return cost*adjust_factor



# transition rules
def eval_transitions(first, second, checks: dict(str, bool)):
    cost = 0
    if (checks["hidden_5th"] and possibility.hiddenFifth(first, second)):
        cost+=transition_costs["hidden_5th"]
    
    if (checks["hidden_8th"] and possibility.hiddenOctave(first, second)):
        cost+=transition_costs["hidden_8th"]
    
    if (checks["parallel_5th"] and possibility.parallelFifths(first, second)):
        cost += transition_costs["parallel_5th"]
    
    if (checks["parallel_8th"] and possibility.parallelOctaves(first, second)):
        cost += transition_costs["parallel_8th"]

    
    return cost



def similarity_eval(realised, original, ngrams_n = 2, show = False):
    realised_progressions = get_chord_progressions(realised)
    original_progressions = get_chord_progressions(original)

    # since we added the lyrics ourselves, we know that lyric number won't go past 1
    r_progression_text = get_text_progressions(realised_progressions.lyrics()[1], has_measures=False)
    og_progression_text = get_text_progressions(original_progressions.lyrics()[1], has_measures=False)

    ngrams_r = ngrams(r_progression_text, ngrams_n)
    ngrams_og = ngrams(og_progression_text, ngrams_n)

    realised_freqdist = FreqDist(ngrams_r)
    original_freqdist = FreqDist(ngrams_og)

    # basically jaccard similarity except impacted by duplicates
    intersection = [ngram for ngram in ngrams_r if ngram in ngrams_og]
    union = ngrams_r + ngrams_n
    jaccard = intersection / union

    # jaro similarity to take into account position of the ngram since music has a temporal element
    jaro = jaro_similarity(ngrams_r, ngrams_og)

    if (show):
        realised_freqdist.plot()
        original_freqdist.plot()

    return {
        "freqdist_r" : realised_freqdist,
        "freqdist_og": original_freqdist,
        "jaccard": jaccard ,
        "jaro": jaro

    }

    # TODO: frequency analysis per bar
    # frequency analysis as a whole - get a proportion/percentage of how much of realised matches original?
    # if show = true shows frequency pplot


# returns a stream of Measures with roman numerals of the chord annotated as lyrics  
def get_chord_progressions(score):
    # get key sig
    analysed_key = score.analyze('key')
    # print(analysed_key.correlationCoefficient)

    chords = score.chordify()

    for c in chords.recurse().getElementsByClass(chord.Chord):
        roman_num = roman.romanNumeralFromChord(c, analysed_key)

        # NOTE: maybe switch to .romanNumeral if .figure is too complex
        c.addLyric(roman_num.figure)


def get_text_progressions(lyrics, has_measures):
    # if has_measures is true, structure of lyrics will be different
    # will return nested loop
    # TODO: convert to text list of progressions -> run nltk on it
    text_lyrics = []

    if (has_measures == False):
        for l in lyrics:
            text_lyrics.append(l.text)

    else:
        # for each loop create equivalent loop of just text equiv
        print("placeholder")

    return text_lyrics



def main():
    print("hi")

    # NOTE: original also has the FB line


    # for now, using pickled objs


    # potenital arguments: maxsemitonelimnit
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