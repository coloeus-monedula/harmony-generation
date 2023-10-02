from xml.etree.ElementTree import Element
from lxml import etree
from copy import deepcopy

def extract_FB(score_path, use_music21_realisation = False):
    score_tree = etree.parse(score_path)
    root = score_tree.getroot()

    # get all parts
    parts = list(root.iter("part"))
    
    # fb is always contained in the last part
    continuo = parts[-1]
    
    # how many divisions in a quaver note
    # divisions = int(bass.xpath('measure[1]/attributes/divisions')[0].text)
    fb = etree.Element("part", id="FB")
    if (use_music21_realisation) :
        # NOTE: assumes bass voice is second to last part
        bass = list(parts[-2].iter("measure"))
        continuo_measures = list(continuo.iter("measure"))
        for i in range(len(bass)):
            measure = combine_bassvoice_and_FB(continuo_measures[i], bass[i])
            fb.append(measure)

        # TODO: add part-list info here

    else:
        for measure in continuo.iter("measure"):
            fb_only = create_FB_measure(measure)
            fb.append(fb_only)
    
    return fb


    # TODO: basically extract this into its own FB xmltree - preserve as much info as possible, minus the x and y positioning

# returns a new measure that contains solely FB information only
# https://stackoverflow.com/questions/4005975/etree-clone-node 
# TODO: make sure measure info is also saved
# steps:
# 1. traverse each immediate subchild. 
# 2. if not figured bass or note, deep clone, add to "fb measure" xml tree
# 3. if figuredbass, add to a temp list
# 4. if Note, check Duration 
# 5. if there is only a single figured bass add that Duration to that figured bass
# 6. add everything in temp list to xml tree

# TODO: if no figured bass before a note, add a REST in fb xml to track duration
def create_FB_measure(measure: Element) -> Element:
    measure_attrib = dict(measure.attrib)
    FB_measure: Element = etree.Element("measure")

    # copy over attributes
    for attrib in measure_attrib:
        FB_measure.set(attrib, measure_attrib.get(attrib))

    
    first_children = measure.xpath("./*")
    temp_fb = []
    for child in first_children:
        if (child.tag == "figured-bass"):
            temp_fb.append(deepcopy(child))
        # copies the note's duration to the fb if not already specified
        # then adds fb to XML tree
        elif (child.tag == "note"):
            duration = child.xpath("duration")[0].text

            if (len(temp_fb) == 1):
                # add duration subelement to fb
                dur_el = etree.SubElement(temp_fb[0], "duration")
                dur_el.text = duration
                FB_measure.append(temp_fb[0])
            elif (len(temp_fb) == 0):
                # inserts a rest with the same duration as the "note" to preserve relative positioning of fb
                # NOTE: THIS ISN'T VALID MUSICXML BUT IS FOR PROCESSING PURPOSES ONLY
                rest = etree.SubElement(FB_measure, "rest")
                dur_el = etree.SubElement(rest, "duration")
                dur_el.text = duration
            else:
                FB_measure.extend(temp_fb)
            
            temp_fb = []
        else:
            FB_measure.append(deepcopy(child))
    
    return FB_measure 


# instead of creating a seperate FB part, combines bass voice part and the FB notations as lyrics
# in order to use the music21 realisations
# since it is 1:1 each FB notation is matched to a bass voice note or a new note is created
# TODO: make sure this part is actual valid musicxml eg. adding part-list stuff to it
def combine_bassvoice_and_FB(continuo, bass):
    # steps:
    # 1. create new measure
    # 2. traverse through continuo part
    # 3. do we need bass? just double continuo notes if there are multiple fbs under it? 
    # 4. also make sure to turn figured bass into lyrics


def main():
    score_path = "chorales/FB_source/musicXML_master/BWV_3.06_FB.musicxml"
    fb = extract_FB(score_path)
    print(etree.tostring(fb, encoding="unicode", pretty_print=True))

if __name__ == "__main__":
    main()