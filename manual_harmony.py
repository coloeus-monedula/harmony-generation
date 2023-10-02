from extract_baseline import extract_FB
from lxml import etree
from music21 import *


# NOTE: assumes a SATB + continuo part that doubles the bass voice - pieces like BWV_248.64_FB unlikely to work
def extract_voices(score_path):

    score = converter.parseFile(score_path)
    parts = score.parts
    # assumes SATB is in order
    # part_list = root.xpath("./part-list[1]")

    # result in dictionary of key (voice) and 
    voices = { 
        "s": parts[0],
        "a": parts[1],
        "t": parts[2],
        "b": parts[3]
    }

    return voices

def main():
    score_path = "chorales/FB_source/musicXML_master/BWV_3.06_FB.musicxml"
    fb = extract_FB(score_path)
    satb = extract_voices(score_path)
    print(satb.get("s").show("text"))
    # print(etree.tostring(fb, encoding="unicode", pretty_print=True))

if __name__ == "__main__":
    main()