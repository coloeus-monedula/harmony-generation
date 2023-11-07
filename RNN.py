import argparse
import muspy
from manual_harmony import convert_music21

# TODO: move this into a separate file later
def preprocess(file):
    # music = muspy.read_musicxml(file)
    # TODO: how to get FBs here? run it through extract baseline - get lyrics separately - rerun with "normal bass"?

    # TO avoid the chord issue mentioned in repo, convert to json ?
    music21_obj = convert_music21(file, return_as_score=True)
    # music21_obj.show()

    music = muspy.from_music21(music21_obj.parts[-1])
    print(music)

# https://github.com/magenta/note-seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--folder","--f", default="chorales/FB_source/musicXML_master/")


    args = parser.parse_args()
    score_path = args.folder + args.file
    preprocess(score_path)



if __name__ == "__main__":
    main()