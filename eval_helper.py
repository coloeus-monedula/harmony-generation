from manual_harmony import manual_parser
from eval import main as eval_score
from postprocessing import muspy_to_music21
from music21 import converter

# https://www.doc.ic.ac.uk/~nuric/coding/argparse-with-multiple-files-to-handle-configuration-in-python.html refactor using this?

# NOTE: if fbs are not added back into the machine generated version, need to turn incomplete off.
def main():
    chord_checks = {
    "close": True,
    "range": True,
    "incomplete": True,
    "crossing": True
    }
    transition_checks = {
    "hidden_5th": True,
    "hidden_8th": True,
    "parallel_5th": True,
    "parallel_8th": True,
    "overlap": True
    }

    max_semitone = 12
    iterations = 5
    to_print = False

    cost_avg = 0
    jaccard_avg = 0
    jaro_avg = 0



    # array of objects
    results = []
    for i in range(iterations):
        returned = manual_parser()
        result = eval_score(chord_checks=chord_checks, transition_checks=transition_checks, max_semitone=max_semitone, scores=returned, to_print=to_print)

        results.append(result)
        jaccard_avg += result["similarity"]["jaccard"]
        jaro_avg += result["similarity"]["jaro"]
        cost_avg +=result["rules"]["total"]

    # pp = pprint.PrettyPrinter()
    # pp.pprint(results)

    jaccard_avg /=iterations
    jaro_avg/=iterations
    cost_avg /=iterations

    print("Average cost for " + str(iterations) + " iterations " + str(round(cost_avg, 4)))
    print("Average jaccard score for " + str(iterations) + " iterations " + str(round(jaccard_avg, 4)))
    print("Average jaro score for " + str(iterations) + " iterations " + str(round(jaro_avg, 4)))


    # file = open("temp/score_objs", "wb")
    # pickle.dump(returned, file)
    # file.close()

    # TODO: predicted melody will also have additional accomp part - do we remove that for this ?


# filename with no .json extension
def machine_eval(filename, og_score):
    chord_checks = {
    "close": True,
    "range": True,
    "incomplete": False,
    "crossing": True
    }
    transition_checks = {
    "hidden_5th": True,
    "hidden_8th": True,
    "parallel_5th": True,
    "parallel_8th": True,
    "overlap": True
    }

    # get both into m21 format
    realised = muspy_to_music21(filename)
    original = converter.parseFile(og_score)

    r_bass = realised.parts[3]
    og_bass = original.parts[3]

    # remove both basses
    original.remove(og_bass)
    realised.remove(r_bass)


    score_objs = {
        "realised": realised,
        "original": original
    }


    result = eval_score(chord_checks=chord_checks, transition_checks=transition_checks, max_semitone=12, scores=score_objs, to_print=True)






if __name__ == "__main__":
    # main()
    machine_eval("b-BWV_36.08_FB.musicxml", "chorales/FB_source/musicXML_master/BWV_36.08_FB.musicxml")