import os
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

def convert_music21(score_path):
    added_fb_xml = extract_FB(score_path, return_whole_scoretree=True, use_music21_realisation=True)

    added_fb_xml = etree.parse(score_path)
    path = "./temp/"+score_path.split("/")[-1]

    if not os.path.exists("./temp"):
        os.makedirs("./temp")

    file = open(path, "wb")
    file.write(etree.tostring(added_fb_xml, pretty_print=True))
    file.close()

    score = converter.parseFile(path)
    # score.parts.show()
    # print(etree.tostring(added_fb_xml))

    bass_fb = score.parts[-1]
    fb = figuredBass.realizer.figuredBassFromStream(bass_fb)

    # bass_fb.show()

    fb_rules = figuredBass.rules.Rules()
    fb_rules.upperPartsMaxSemitoneSeparation = None
    fb_rules.partMovementLimits = [(1, 2), (2, 12), (3, 12)]
    realisation = fb.realize()

    realisation.generateRandomRealization().show()


def main():
    # http://www.continuo.ca/files/Figured%20bass%20chart.pdf figured bass cheatsheet

    score_path = "chorales/FB_source/musicXML_master/BWV_3.06_FB.musicxml"
    # score_path = "chorales/FB_source/musicXML_master/BWV_470_FB.musicxml"
    # fb = extract_FB(score_path)
    # satb = extract_voices(score_path)
    convert_music21(score_path)
    
    # print(satb.get("s").show("text"))
    # print(etree.tostring(fb, encoding="unicode", pretty_print=True))

    # added_fb_xml = extract_FB(score_path, return_whole_scoretree=True, use_music21_realisation=True)

if __name__ == "__main__":
    main()