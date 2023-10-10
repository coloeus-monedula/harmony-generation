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
    parts = score.parts


    voices = {
        "s": parts[0],
        "a": parts[1],
        "t": parts[2],
        "b": parts[3],
        "fb": parts[-1]
    }

    # voices["s"].show()
    return voices



#generates a harpsichord realisation below SATB, like how FB is typically used
# def fb_realisation_harpsichord():


def fb_realisation_satb(voices):
    bass_fb = voices["fb"]
    fb = figuredBass.realizer.figuredBassFromStream(bass_fb)

    # changing rules
    fb_rules = figuredBass.rules.Rules()
    fb_rules.upperPartsMaxSemitoneSeparation = None
    fb_rules.partMovementLimits = [ (1,4), (2, 12), (3, 12)]
    fb_rules.applyConsecutivePossibRulesToResolution = True
    fb_rules.applySinglePossibRulesToResolution = True

    # editing original soprano part
    soprano = voices["s"]
    incomplete_bar = soprano.measure(0)
    if (incomplete_bar is not None):
        # increases each Note's offset in the first bar so rests are on left side
        offset = incomplete_bar.paddingLeft
        for n in incomplete_bar.notes:
            n.offset = n.offset + offset

        # increase measure number to 1 if the first bar is incomplete, to match w realised score
        for m in soprano.getElementsByClass("Measure"):
            new_num = m.number + 1
            m.number = new_num

    

    # have to do 4 parts, else 0 solutions
    realisation = fb.realize(numParts=4)
    realisation.keyboardStyleOutput = False

    realised_score = realisation.generateRandomRealization()
    # add voice
    for part in realised_score.parts:
        part.insert(0, instrument.Choir())



    realised_soprano = realised_score.parts[0]
    realised_score.replace(realised_soprano, soprano)

    # print(realised_score.parts[-1].measure(1).paddingLeft)
    realised_score.show()
    # TODO: make so it's only "tenor" and "alto" by doing some sort of rule, then splice soprano part together
    # TODO: make it so that alto and tenor CAN'T BE greater than soprano. 
    # maxPitch = lowest soprano note

def main():
    # http://www.continuo.ca/files/Figured%20bass%20chart.pdf figured bass cheatsheet

    score_path = "chorales/FB_source/musicXML_master/BWV_3.06_FB.musicxml"
    # score_path = "chorales/FB_source/musicXML_master/BWV_470_FB.musicxml"
    # fb = extract_FB(score_path)
    # satb = extract_voices(score_path)
    voices = convert_music21(score_path)
    fb_realisation_satb(voices)
    
    # print(satb.get("s").show("text"))
    # print(etree.tostring(fb, encoding="unicode", pretty_print=True))

    # added_fb_xml = extract_FB(score_path, return_whole_scoretree=True, use_music21_realisation=True)

if __name__ == "__main__":
    main()