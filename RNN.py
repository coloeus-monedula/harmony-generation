import argparse
import muspy
from manual_harmony import convert_music21


def build_model():
    pass


def train():
    pass


def test():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    # default should be wherever the translated musicxml files are
    parser.add_argument("--folder","--f", default="chorales/FB_source/musicXML_master/")


    args = parser.parse_args()
    score_path = args.folder + args.file



if __name__ == "__main__":
    main()