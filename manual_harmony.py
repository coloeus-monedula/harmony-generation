import os
from extract_baseline import extract_FB
from lxml import etree
from music21 import *
import argparse


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


def fb_realisation_satb(voices, melody, maxpitch, score_parts, rules_args = None):
    bass_fb = voices["fb"]
    fb = figuredBass.realizer.figuredBassFromStream(bass_fb)

    # changing rules
    # http://www.choraleguide.com/vl-spacing.php ?
    fb_rules = figuredBass.rules.Rules()

    # typically supposed to have this be 12 but this with maxPitch often makes 0 realisations
    fb_rules.upperPartsMaxSemitoneSeparation = None
    
    fb_rules.partMovementLimits = [ (1,5), (2, 14), (3, 14)]

    # more restrictive
    # fb_rules.partMovementLimits = [ (1,4), (2, 12), (3, 12)]
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

    # find highest pitch in soprano part as we have to generate all satb parts
    # and hope that alto and tenor don't cross soprano part (which in practice alto part does)
    # lowestPitch = sorted(soprano.pitches)[0]
    highestPitch = sorted(soprano.pitches)[-1]

    # have to do 4 parts, else 0 solutions
    # NOTE: if no realisations loosen upperMaxSemitone then partMovementLimits, then others 
    realisation = fb.realize( maxPitch=highestPitch, fbRules=fb_rules)
 
    realisation.keyboardStyleOutput = False

    # TODO: have check for if 0 realisations - gradually loosen rules until solution found
    # TODO: and/or transpose soprano an octave up?
    realised_score = realisation.generateRandomRealization()
    # add voice
    for part in realised_score.parts:
        part.insert(0, instrument.Choir())



    realised_soprano = realised_score.parts[0]

    # put at the top
    realised_score.insert(-1, soprano)
    realised_score.remove(realised_soprano)

    realised_score.show()

    # TODO: replace with old bass


    # TODO: ARGUMENTS INCLUDE: score file name, score folder(defaults to chorales etc.), which SAT part is considered "melody", 
    # original parts to insert (other than melod) narg --r for replace , --compare keep og. melody part by default is replaced at the very least
    # add rules true/false, maxpitch (default highest soprano pitch), all of the rule adjustments only if there isn't a --no-rules flag

def main():
    parser = argparse.ArgumentParser(description="Realise harmony for a SATB + intrument baseline Bach Chorale using Music21 figured bass harmony rules.")
    parser.add_argument("file")
    parser.add_argument("--folder","--f", default="chorales/FB_source/musicXML_master/")
    parser.add_argument("--melody", "--m", default="s", nargs = 1, choices=["s","a","t"], type=str.lower, help="Which part (Soprano, Alto, Tenor) should be considered melody.")
    parser.add_argument("--replace","--r", nargs="*", choices=["s","a","t"],  type=str.lower, help = "Which realised parts (Soprano, Alto, Tenor) should be replaced with original parts. The melody line is always replaced, unless --compare is specified. Takes priority over --compare. " )
    parser.add_argument("--compare","--c", nargs="*",choices=["s","a","t"],  type=str.lower, help="Which realised parts (Soprano, Alto, Tenor) should have their original part on the score as comparison. ")
    parser.add_argument("--maxpitch", "--mp", default="s", help = "Upper limit on highest pitch realisation will reach.")
    parser.add_argument("--no-rules","--nr", action= "store_true", help = "If specified, doesn't apply a Rules object to the realisation.")

    rules = parser.add_argument_group("rules")
    rules.add_argument("--parts-sep", "--ps", default=0, type=int, help = "Maximum amount of semitones apart the upper parts of the realisation (here everything except bass) can be. Default is None (0) ie. no limitations. ")
    rules.add_argument("--no-consec-rules", "--ncr", action="store_false", help="Doesn't apply consecutive possibility rules to possible realisations. ")
    rules.add_argument("--no-single-rules", "--nsr", action = "store_false", help = "Doesn't apply single possibility rules to possible realisations. ")

    # wrap in tuple individually
    rules.add_argument("--part-move-limit", "--part-limit", "--pl", "--pml", nargs=2, action="append", help = "Set maximum amount of semitones a part's pitch can move to for the next note. First number is partNumber from highest part (soprano) to lowest (bass), second number is maximum semitone separation. Not specifying sets limits to [ (1,5), (2, 14), (3, 14)].", type=int)


    args = parser.parse_args()
    # http://www.continuo.ca/files/Figured%20bass%20chart.pdf figured bass cheatsheet

    score_path = args.folder + args.file

    rules_args = None
    # adds rules to a dict
    if (args["no-rules"] == False):
        rules_args = {}

        if args["parts-sep"] == 0:
            rules_args.separation = None
        else:
            rules_args.separation = args["parts-sep"]

        rules_args.consecutive = args["ncr"]
        rules_args.single = args["nsr"]

        if args["part-limit"] is None:
            rules_args.move_lim = [(1,5), (2, 14), (3, 14)]
        else:
            move_lim = []
            for pair in args["part-limit"]:
                move_lim.append(tuple(pair))
            
            rules_args.move_lim = move_lim

    # TODO: check if melody line is in compare. if so, add to compare. if not, add to replace
    # then just add as is

    score_parts = {
        "replace" : [],
        "compare" : []
    }
    melody = args.melody
    if (melody not in args["compare"]):
        score_parts["replace"].append(melody)
    
    score_parts["replace"].extend(args["replace"])
    score_parts["compare"].extend(args["compare"])

            

    # score_path = "chorales/FB_source/musicXML_master/BWV_470_FB.musicxml"
    # score_path = "chorales/FB_source/musicXML_master/BWV_3.06_FB.musicxml"
    # fb = extract_FB(score_path)
    # satb = extract_voices(score_path)
    voices = convert_music21(score_path)
    fb_realisation_satb(voices)
    
    # print(satb.get("s").show("text"))
    # print(etree.tostring(fb, encoding="unicode", pretty_print=True))

    # added_fb_xml = extract_FB(score_path, return_whole_scoretree=True, use_music21_realisation=True)

if __name__ == "__main__":
    main()