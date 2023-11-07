import argparse
from music21.figuredBass import possibility, realizerScale
from music21 import chord, note
from music21 import *
from nltk import ngrams, FreqDist
from nltk.metrics.distance import jaro_similarity
import dill as pickle
import pprint 


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
# http://www.choraleguide.com/vl-spacing.php voice crossing sometimes done to avoid parallels - implies that cost should be lower than parallel costs 
# but given it's generally to avoid can't be TOO much lower
# TODO: add "
chord_costs = {
    # assume to fall under "smooth connection of each part" metric
    "close": -4,
    "doubled_leading": 1,
    # checks for each voice, up to 4 per chord in total
    "not_in_range": 1,
    # ie. figure bass realisations not in realised - weighted strongly
    "incomplete": 5,
    "crossing": 1.5
}

# TODO: thresholding to see what is determined as "smooth connection"?
# parallel fifth is successive fifth, hidden _ is a subcategory of successive _s hence the 3 weighting
# NOTE: voice overlap is kind of the "beginning" of a voice cross so is weighted similarity
transition_costs = { 
    "parallel_8th": 5,
    "parallel_5th": 3,
    "hidden_5th": 3,
    "hidden_8th": 3,
    "overlap": 1.5


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

        # print("chord:", aChord.lyrics)
        if (format == "obj"):
            pitches_list.append(pitches)
        elif (format =="number"):
            midi_pitches = (pitch.midi for pitch in pitches)
            pitches_list.append(midi_pitches)

    return pitches_list


    
# negative rules increase cost bc less likely
# https://web.mit.edu/music21/doc/moduleReference/moduleChord.html#chord also use the Chord() funcs

# http://web.mit.edu/music21/doc/usersGuide/usersGuide_09_chordify.html chordify

def rules_based_eval(score, chord_checks, trans_checks, local_adjust = 1, trans_adjust = 1):
    total_costs = 0
    local_list = []
    trans_list = []
    local_total = 0
    trans_total = 0

    # "If a Score or Part of Measures is provided, a Stream of Measures will be returned"
    # remove redundant false to account for the fact sometimes voices sing the same notes
    chords = score.chordify(addPartIdAsGroup = True, removeRedundantPitches = False)

    fb = score.parts[-1]
    analysed_key = score.analyze('key')
    print("Analysed key of", analysed_key, "with correlation coefficient of", round(analysed_key.correlationCoefficient, 4))


    # make measure offsets for each score equal to each other so offset matching works
    # NOTE: assumes the chordified and normal parts should have equal number of measures
    fb_measures = fb.getElementsByClass(stream.Measure)
    chords_measures = chords.getElementsByClass(stream.Measure)
    for i in range(0,len(fb_measures)):
        # match chordify measure offset to fb's value
        fb_offset = fb_measures[i].offset
        chords_measures[i].offset = fb_offset

    
    #1:1 fb notation to chords, either None or in notationString format 
    fb_list = []
    # https://groups.google.com/g/music21list/c/pr8616w2bT0/m/90AatjqsAQAJ
    # adding lyrics to chordify function based off this one 
    for aChord in chords.flatten():
        chord_offset = aChord.offset

        for melody_note in fb.flatten().getElementsByOffset(chord_offset):
            try:
                lyrics = melody_note.lyrics
                # exclude lyrics that dont have FB notation
                if lyrics[0].rawText.strip() == "":
                    fb_list.append(None)
                else:
                    notation_str = []
                    for lyric in lyrics:
                        notation_str.append(lyric.rawText)
                        # also adds to chords as lyrics in case it's needed in the future
                        aChord.addLyric(lyric.rawText, lyric.number)
                    fb_list.append(",".join(notation_str))

            except (AttributeError):
                continue

    # print(fb_list)
    pitches = get_pitches_music21_chords(chords, format="obj")
    size = len(pitches)

    # everything up to and excluding the n-1 chord
    for i in range(0, size - 1):
        first = pitches[i]
        second = pitches[i+1]

        local_cost = eval_chord(first, chord_checks, local_adjust, fb_list[i], analysed_key)
        transition_cost = trans_adjust * eval_transitions(first, second, trans_checks)

        local_total+=local_cost
        trans_total += transition_cost

        total = local_cost+transition_cost
        total_costs += total
        local_list.append(local_cost)
        trans_list.append(transition_cost)

    
    # n - 1 chord
    first = pitches[size - 2]
    second = pitches[size - 1]
    local_cost = eval_chord(first, chord_checks, local_adjust, fb_list[size-2],  analysed_key) + eval_chord(second, chord_checks, local_adjust, fb_list[size-1],  analysed_key)
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
def eval_chord(chord, checks: dict, adjust_factor, fb, analyzed_key):

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
        if ( a < vocal_ranges["a"][0] or a > vocal_ranges["a"][1]):
            cost +=chord_costs["not_in_range"]
        if ( t < vocal_ranges["t"][0] or t > vocal_ranges["t"][1]):
            cost +=chord_costs["not_in_range"]
        if ( b < vocal_ranges["b"][0] or b > vocal_ranges["b"][1]):
            cost +=chord_costs["not_in_range"]


    # TODO: do check if doubled leading note ? - do they mean the 7th or just the semitoneness? 

    # NOTE: currently only checks if there are explicit FB notations on the screen
    if (checks["incomplete"] and fb is not None) :
        (s, a, t, b) = chord

        # what the program thinks the realised piece's keysig is - may not be the original's
        tonic = analyzed_key.tonic.name
        key_scale = analyzed_key.mode
        # print(tonic, key_scale)
        scale = realizerScale.FiguredBassScale(tonic, key_scale)
        pitches_to_include = scale.getPitchNames(b.nameWithOctave, fb)
        # print(pitches_to_include)
        if possibility.isIncomplete(chord, pitches_to_include):
            cost += chord_costs["incomplete"]


    if (checks["crossing"] and possibility.voiceCrossing(chord)):
        cost += chord_costs["crossing"]

    return cost*adjust_factor



# transition rules
def eval_transitions(first, second, checks: dict):
    cost = 0
    if (checks["hidden_5th"] and possibility.hiddenFifth(first, second)):
        cost+=transition_costs["hidden_5th"]
    
    if (checks["hidden_8th"] and possibility.hiddenOctave(first, second)):
        cost+=transition_costs["hidden_8th"]
    
    if (checks["parallel_5th"] and possibility.parallelFifths(first, second)):
        cost += transition_costs["parallel_5th"]
    
    if (checks["parallel_8th"] and possibility.parallelOctaves(first, second)):
        cost += transition_costs["parallel_8th"]

    if (checks["overlap"] and possibility.voiceOverlap(first, second)):
        cost +=transition_costs["overlap"]
    
    return cost



def similarity_eval(realised, original, ngrams_n = 2, show = False):
    realised_progressions = get_chord_progressions(realised)
    original_progressions = get_chord_progressions(original)

    # since we added the lyrics ourselves, we know that lyric number won't go past 1
    r_progression_text = get_text_progressions(realised_progressions.lyrics()[1], has_measures=False)
    og_progression_text = get_text_progressions(original_progressions.lyrics()[1], has_measures=False)

    ngrams_r = list(ngrams(r_progression_text, ngrams_n))
    ngrams_og = list(ngrams(og_progression_text, ngrams_n))

    realised_freqdist = FreqDist(ngrams_r)
    original_freqdist = FreqDist(ngrams_og)

    # basically jaccard similarity except impacted by duplicates
    intersection = [ngram for ngram in ngrams_r if ngram in ngrams_og]
    union = ngrams_r + ngrams_og
    jaccard = len(intersection) / len(union)

    # jaro similarity to take into account position of the ngram since music has a temporal element
    jaro = jaro_similarity(ngrams_r, ngrams_og)

    if (show):
        realised_freqdist.plot()
        original_freqdist.plot()

    return {
        "freqdist_r" : realised_freqdist,
        "freqdist_og": original_freqdist,
        "jaccard": round(jaccard, 4) ,
        "jaro": round(jaro, 4)

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

        # NOTE: switched to .romanNumeral since .figure gives too detailed chords numberings for similarity evals
        c.addLyric(roman_num.romanNumeral)

    return chords


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
        # TODO
        print("placeholder")

    return text_lyrics



# checks are false by default, turn on by params
def main(standalone = False, chord_checks = {
    "close": False,
    "range": False,
    "incomplete": False,
    "crossing": False
}, transition_checks = {
    "hidden_5th": False,
    "hidden_8th": False,
    "parallel_5th": False,
    "parallel_8th": False,
    "overlap": False
}, max_semitone = 12, scores = None, to_print=False):

    global max_semitone_separation
    # ie. chord and transition checks aren't passed in via another python program and is via argparse
    if (standalone) :
        parser = argparse.ArgumentParser(description = "Evaluates a realised SATB choral harmony for a single pickled file (from manual_harmony.py) containing realised and original music21 objects.")
        parser.add_argument("file", default="temp/score_objs", nargs="?")
        parser.add_argument("--all", action="store_true", help="Turns on all evaluation checks.")
        parser.add_argument("--max-semitone", "--mss",type=int, default=12, help="Maximum semitone separation to differentiate what is considered close vs open harmony. Defaults to 12.")
        parser.add_argument("--print", action="store_true")
        
        chord = parser.add_argument_group("chord checks")
        chord.add_argument("--close", action="store_true", help="Turn on close harmony eval checks.")
        chord.add_argument("--range", action="store_true", help="Turn on on in vocal range eval checks.")
        chord.add_argument("--incomplete", action="store_true", help = "Turn on incomplete FB realisation checks.")
        chord.add_argument("--crossing", action="store_true", help = "Turn on voice crossing realisation checks.")

        transition = parser.add_argument_group("transition checks")
        transition.add_argument("--parallel8", action="store_true", help="Turn on parallel octave eval checks.")
        transition.add_argument("--parallel5", action="store_true", help="Turn on parallel 5th eval checks.")        
        transition.add_argument("--hidden8", action="store_true", help="Turn on hidden octave eval checks.")
        transition.add_argument("--hidden5", action="store_true", help="Turn on hidden 5th eval checks.")
        transition.add_argument("--overlap", action="store_true", help="Turn on voice overlap eval checks.")
        args = parser.parse_args()


        with open(args.file, "rb") as f:
            scores = pickle.load(f)

        if (args.all == True):
            chord_checks["close"] = True
            chord_checks["range"] = True
            chord_checks["incomplete"] = True
            chord_checks["crossing"] = True
            transition_checks["hidden_5th"]= True
            transition_checks["hidden_8th"] = True
            transition_checks["parallel_5th"] = True
            transition_checks["parallel_8th"] = True
            transition_checks["overlap"] = True


        if (args.close):
            chord_checks["close"] = True
        if (args.range):
            chord_checks["range"] = True
        if (args.incomplete):
            chord_checks["incomplete"] = True
        if (args.crossing):
            chord_checks["crossing"] = True
        if (args.hidden5):
            transition_checks["hidden_5th"]= True
        if (args.hidden8):
            transition_checks["hidden_8th"] = True
        if (args.parallel5):
            transition_checks["parallel_5th"] = True
        if (args.parallel8):
            transition_checks["parallel_8th"] = True
        if (args.overlap):
            transition_checks["overlap"] = True

        max_semitone_separation = args.max_semitone

    else:
        max_semitone_separation = max_semitone


    realised = scores["realised"]
    # realised.show()

    # reconstructing original score
    # have to do this for both
    original = stream.Score()
    parts = list(scores["original"].values())
    for p in parts:
        original.insert(p)

    # original.show()

    rules_results = rules_based_eval(realised, chord_checks, transition_checks)


    similarity_results = similarity_eval(realised, original)

    if (standalone and args.print):
        print(rules_results)
        pp = pprint.PrettyPrinter()
        pp.pprint(similarity_results)

    return {
        "rules": rules_results,
        "similarity": similarity_results
    }

if __name__ == "__main__":
    main(standalone=True)