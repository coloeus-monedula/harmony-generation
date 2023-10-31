from manual_harmony import manual_parser
import dill as pickle

# https://www.doc.ic.ac.uk/~nuric/coding/argparse-with-multiple-files-to-handle-configuration-in-python.html refactor using this?
def main():

    returned = manual_parser()
    # at the moment just pickles thigns

    file = open("temp/score_objs", "wb")
    pickle.dump(returned, file)
    file.close()



if __name__ == "__main__":
    main()