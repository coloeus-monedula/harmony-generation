from os import path
import pickle
from manual_harmony import realise
from eval import main as eval_score
from postprocessing import muspy_to_music21, tensor_to_json
from music21 import converter
from machine_harmony import eval_model


def eval_one(og_score, args_dict, generate_type,chord_checks, transition_checks, save = False, results_file = "",     remove_add_dict = {
        "remove": ["s"],
        "add": ["s"]
    }):


    max_semitone = 12
    iterations = 5
    to_print = False

    cost_avg = 0
    jaccard_avg = 0
    jaro_avg = 0
    test_accuracy = 0

    # array of objects
    results = []
    for i in range(iterations):
        is_ML = False
        if generate_type == "m21":
            score_objs = realise(og_score, args_dict, remove_add_dict)
        else:
            is_ML = True

            score_name = args_dict["score_name"]
            model_path = args_dict["model_path"]
            token_path = args_dict["token_path"]
            test_file = args_dict["test_file"]
            JSON_folder = args_dict["JSON_folder"]
            original = args_dict["original"]
            realised, test_acc = get_ML_generated(score_name, model_path, token_path, test_file, JSON_folder)

            score_objs = {
                "realised": realised,
                "original":original
            }

            test_accuracy+=test_acc

        result = eval_score(chord_checks=chord_checks, transition_checks=transition_checks, max_semitone=max_semitone, scores=score_objs, to_print=to_print, is_ML=is_ML)

        results.append(result)
        jaccard_avg += result["similarity"]["jaccard"]
        jaro_avg += result["similarity"]["jaro"]
        cost_avg +=result["rules"]["total"]

    # pp = pprint.PrettyPrinter()
    # pp.pprint(results)

    jaccard_avg /=iterations
    jaro_avg/=iterations
    cost_avg /=iterations

    if (is_ML):
        test_accuracy /=iterations
        print("Average test accuracy for "+ str(iterations) +" iterations " + str(round(test_acc, 4)))
    print("Average cost for " + str(iterations) + " iterations " + str(round(cost_avg, 4)))
    print("Average jaccard score for " + str(iterations) + " iterations " + str(round(jaccard_avg, 4)))
    print("Average jaro score for " + str(iterations) + " iterations " + str(round(jaro_avg, 4)))

    if save:
        file = open(results_file, "wb")
        pickle.dump(results, file)
        file.close()

    # TODO: predicted melody will also have additional accomp part - do we remove that for this ?

 # remove_add_dict =  parameters for music21 realisation. does default of replacing the soprano part with original chorale's soprano

def eval_all_variations(score_num, rules_args, chord_checks, transition_checks,ML_args,
    remove_add_dict = {
        "remove": ["s"],
        "add": ["s"]
    }, save = False):
    # filename, og_score, rules_args
    og_score =  "chorales/FB_source/musicXML_master/BWV_{}_FB.musicxml".format(score_num)
    filename = "BWV_{}_FB.musicxml".format(score_num)

    # uni_file = "u-BWV_{}_FB.musicxml".format(score_num)
    # the predictions for the trained model will currently always be the same, due to the lack of randomness during prediction (as we feed in parts from the score)
    # TODO: REMOVE
    # bi_scores = convert_ML_to_m21(bi_file, og_score)
    # uni_scores = convert_ML_to_m21(uni_file, og_score)
    # realised_scores = realise(og_score, rules_args, remove_add_dict)

    # original score for use with ML eval
    original = converter.parseFile(og_score)
    og_accomp = original.parts[-1]
    original.remove(og_accomp)

    # results_file = "temp/"+bi_file+"_eval.pkl"
    # print("\nRealised harmony using Music21 module.")
    # eval_one(og_score, rules_args,"m21", chord_checks=chord_checks, transition_checks=transition_checks,save=save, results_file="artifacts/r-BWV_"+score_num+"_FB_eval.pkl", remove_add_dict=remove_add_dict)

    bi_args = {
        "score_name":filename,
        "model_path":ML_args["b_model_path"],
        "token_path":ML_args["token_path"],
        "test_file":ML_args["test_file"],
        "JSON_folder":ML_args["JSON_folder"],
        "original": original

    }

    print("\nGenerated harmony - bidirectional.")
    eval_one(og_score, bi_args,"bi",chord_checks=chord_checks, transition_checks=transition_checks, save=save, results_file="artifacts/b-"+filename+"_eval.pkl")

    uni_args = {
        "score_name":filename,
        "model_path":ML_args["b_model_path"],
        "token_path":ML_args["token_path"],
        "test_file":ML_args["test_file"],
        "JSON_folder":ML_args["JSON_folder"],
        "original": original

    }

    print("\nGenerated harmony - unidirectional.")
    eval_one(og_score, uni_args,"uni",chord_checks=chord_checks, transition_checks=transition_checks, save=save, results_file="artifacts/u-"+filename+"_eval.pkl")


# returns ML generation in music21 format
# covers prediction + postprocessing, cutting out unnecessary code eg. saving to file
def get_ML_generated(score_name, model_path, token_path, test_file, JSON_folder = "generated_JSON"):
    # use eval_model
    print(score_name)

    # these are constant
    params = {
        "output_num": 6,
        "resolution": 8
    }
    accuracy, generated = eval_model(model_path, token_path, True, test_file, params, single_file_name=score_name )

    basename = path.splitext(score_name)[0]
    tensor_to_json(generated, JSON_folder, "eval_"+basename+".json", token_path=token_path)
    realised = muspy_to_music21("eval_"+basename, json_folder=JSON_folder)

    # clean up
    r_accomp = realised.parts[-1]
    realised.remove(r_accomp)

    # make realised's Parts offsets to 0
    for part in realised.parts:
        part.offset = 0

    return realised, accuracy
    


# # filename with no .json extension
# # NOTE: requires generated JSON artifacts from "postprocessing.py"
# def convert_ML_to_m21(filename, og_score):

#     # get both into m21 format
#     #NOTE: the predictions for the trained model will currently always be the same, due to the lack of randomness during prediction (as we feed in parts from the score)
#     realised = muspy_to_music21(filename)
#     original = converter.parseFile(og_score)

#     r_accomp = realised.parts[-1]
#     og_accomp = original.parts[-1]

#     # remove both accompaniments
#     original.remove(og_accomp)
#     realised.remove(r_accomp)


#     score_objs = {
#         "realised": realised,
#         "original": original
#     }

#     # make realised's Parts offsets to 0
#     for part in realised.parts:
#         part.offset = 0

#     # transpose accomp an octave up, to match with how music21 analyses the realised and original scores
#     # since music21 looks at the notes on the score and ignores the fact the actual pitch is an octave lower, but muspy didn't and wrote the notes at their actual pitch
#     # realised.parts[-1].transpose("P8", inPlace=True)


#     return score_objs



default = {
        "separation":None,
        "consecutive": False,
        "single":False,
        "parallel5": True,
        "parallel8": True,
        "hidden5": True,
        "hidden8": True,
        "crossing": True,
        "overlap": True,
        "incomplete": True,
        "move_lim": [(1,5), (2, 14), (3, 14)]
    }

chord_checks = {
"close": True,
"range": True,
"incomplete": False, #NOTE: this is somewhat unreliable - see rules_based_eval() for notes
"crossing": True
}

transition_checks = {
"hidden_5th": True,
"hidden_8th": True,
"parallel_5th": True,
"parallel_8th": True,
"overlap": True
}

if __name__ == "__main__":
    # main()
    options = ["36.08", "145.05", "245.14"]

    # does defaults from manual_harmony, with a few modifications
    rules_args = {"36.08": {
        "separation":None,
        "consecutive": False,
        "single":False,
        "parallel5": True,
        "parallel8": True,
        "hidden5": True,
        "hidden8": True,
        "crossing": True,
        "overlap": False,
        "incomplete": False,
        "move_lim": [(1,5), (2, 14), (3, 14)]
    },
    "145.05":  {
        "separation":None,
        "consecutive": True,
        "single":True,
        "parallel5": True,
        "parallel8": True,
        "hidden5": True,
        "hidden8": True,
        "crossing": True,
        "overlap": True,
        "incomplete": True,
        "move_lim": [(1,5), (2, 14), (3, 14)]
    },
    "245.14":  {
        "separation":None,
        "consecutive": False,
        "single":True,
        "parallel5": True,
        "parallel8": True,
        "hidden5": True,
        "hidden8": True,
        "crossing": True,
        "overlap": True,
        "incomplete": True,
        "move_lim": [(1,5), (2, 14), (3, 14)]
    }}

    
    ML_args = {
        "b_model_path":"artifacts/bi-l-230.pt",
        "u_model_path":"artifacts/uni-l-230.pt",
        "token_path":"artifacts/230_tokens.pkl",
        "test_file":"artifacts/230_preprocessed_test.pt",
        "JSON_folder":"generated_JSON",

    }

    for i in range(len(options)):
        score_num = options[i]
        rules = rules_args.get(score_num)
        print("\nResults for {}".format(score_num))
        eval_all_variations(score_num, rules_args = rules, chord_checks=chord_checks, transition_checks=transition_checks, ML_args = ML_args,save = False)