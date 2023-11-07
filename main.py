import pprint
from manual_harmony import manual_parser
from eval import main as eval_score

# https://www.doc.ic.ac.uk/~nuric/coding/argparse-with-multiple-files-to-handle-configuration-in-python.html refactor using this?
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



if __name__ == "__main__":
    main()