from xml.etree.ElementTree import Element
from lxml import etree
from copy import deepcopy
import math


# modifiers dictionary taken from 
# https://github.com/cuthbertLab/music21/blob/master/music21/figuredBass/notation.py 
# modifiersDictXmlToM21 = {
#     'sharp': '#',
#     'flat': 'b',
#     'natural': '\u266e',
#     'double-sharp': '##',
#     'flat-flat': 'bb',
#     'backslash': '\\',
#     'slash': '/',
#     'cross': '+'
# }

# modifier information taken from https://web.mit.edu/music21/doc/moduleReference/moduleFiguredBassNotation.html
# only accepts flats, sharps, naturals, double flats and double sharps
# NOTE: dataset encodes forward slash = lowered intervals. backslash = raised intervals according to corresponding paper
# NOTE: cross appears to be allowed as a suffix but will become prefixed - translate to sharp instead if realisation will be affected by that?
XMLToFBModifiers = {
    'sharp': '#',
    'flat': 'b',
    'natural': 'n',
    'double-sharp': '##',
    'flat-flat': 'bb',
    'backslash': '#',
    'slash': 'b',
    'cross': '+'
}

# names taken from https://w3c.github.io/musicxml/musicxml-reference/data-types/note-type-value/
# amount of division each note takes up when division = 1 (ie. crotchet is 1)
# 2 d.p at the moment
standardNoteTypeValue = {
    0.03: "128th",
    0.06: "64th",
    0.13: "32nd",
    0.25: "16th",
    0.5 : "eighth",
    1: "quarter",
    2: "half",
    4: "whole",
    8: "breve",
    16: "long",
    32: "maxima"
}


# remove comparison removes the original continuo part
def extract_FB(score_path, use_music21_realisation = False, return_whole_scoretree = False, combine_parts = False, remove_OG_accomp = False):
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
        # changed this to divisions of FB part since spliced together Fb-bass part has different divisions
        divisions = int(parts[-1].xpath('measure[1]/attributes/divisions')[0].text)
        bass = list(parts[-2].iter("measure"))
        continuo_measures = list(continuo.iter("measure"))
        for i in range(len(bass)):
            if (combine_parts):
                measure = combine_two_parts(continuo_measures[i], bass[i], divisions)
            else:
                measure = convert_one_part(continuo_measures[i], divisions)
            fb.append(measure)

        #add part-list info here
        fb_scorepart = etree.Element("score-part", id="FB")
        if (combine_parts):
            etree.SubElement(fb_scorepart, "part-name").text = "Bass and FB"
        else:
            etree.SubElement(fb_scorepart, "part-name").text = "Converted FB"

        part_list = root.xpath("./part-list")[0]
        continuo_scorepart = part_list[-1]
        if remove_OG_accomp:
            part_list.replace(continuo_scorepart, fb_scorepart)
        else:
            part_list.append(fb_scorepart)


    else:
        for measure in continuo.iter("measure"):
            fb_only = create_FB_measure(measure)
            fb.append(fb_only)
    
    if (return_whole_scoretree):
        if remove_OG_accomp:
            root.replace(continuo, fb)
        else:
            root.append(fb)
        # NOTE: scoretree returned if not doing music21 realisation is not "proper" MusicXML
        return score_tree
    else:
        return fb



# returns a new measure that contains solely FB information only
# steps:
# 1. traverse each immediate subchild. 
# 2. if not figured bass or note, deep clone, add to "fb measure" xml tree
# 3. if figuredbass, add to a temp list
# 4. if Note, check Duration 
# 5. if there is only a single figured bass add that Duration to that figured bass
# 6. add everything in temp list to xml tree
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



# traverse through fb
# add figured bass to temp fb list
# add everything else to measure
# used since we're not combining bass voice part and accompaniment's FB notations anymore
def convert_one_part(part: Element, divisions: int) -> Element:
    return combine_two_parts(part, part, divisions)

# instead of creating a seperate FB part, combines bass voice part and the FB notations as lyrics
# in order to use the music21 realisations
# since it is 1:1 each FB notation is matched to a bass voice note or a new note is created
# NOTE: assumes bass and continuo have the same notes albeit transposed an octave.
def combine_two_parts(continuo: Element, bass: Element, divisions: int) -> Element:
    bass_attrib = dict(bass.attrib)

    FB_measure:Element = etree.Element("measure")

    for attrib in bass_attrib:
        FB_measure.set(attrib, bass_attrib.get(attrib))

    # traverse through bass and continuo part at same time
    continuo_children = continuo.xpath("./*")
    bass_children = bass.xpath("./*")
    bass_child: Element = bass_children[0]
    temp_fb = []
    continued_FB = None

    # assuming bass and continuo have same notes continuo will always be equal or longer due to also having FB too 
    for child in continuo_children:
        if (child.tag == "figured-bass"):
            # 2D array
            (lyrics, continued_FB) = turn_FBxml_into_lyrics(child, continued_FB)
            temp_fb.append(lyrics)
        elif (child.tag == "note"):
            # advances bass to the next <note>. anything that isn't <note> gets appended to FB measure
            while (bass_child is not None and bass_child.tag != "note"):
                FB_measure.append(deepcopy(bass_child))
                bass_child = bass_child.getnext()

            # NOTE: this happens when there is note mismatch between continuo and bass - shows up at the end of a bar
            # print(etree.tostring(child, encoding="unicode", pretty_print=True))
            if bass_child is None:
                print("Error: Bass child is None - suggests a unresolved mismatch between bass and accompaniment part.")
                continue

            if len(temp_fb) == 1:
                # adds lyrics 
                lyrics = temp_fb[0]
                fb_bass = append_lyrics_to_bass(deepcopy(bass_child), lyrics)
                FB_measure.append(fb_bass)
                
            elif len(temp_fb) == 0:
                FB_measure.append(deepcopy(bass_child))
            else: 
                # creates new note using each fb's duration
                for fb in temp_fb:
                    fb_bass = create_new_bassnote(fb, bass_child, divisions)
                    FB_measure.append(fb_bass)

            # move onto next bass_child after being processed
            bass_child = bass_child.getnext()

            temp_fb = []

    return FB_measure


def append_lyrics_to_bass(fb_bass, lyrics):
    for lyric in lyrics:
        fb_bass.append(lyric)
    return fb_bass

def create_new_bassnote(fb: Element, bass_child: Element, divisions):
    # fetch duration value from first <fb> and add to copied <note>
    # then remove it from <fb>
    new_duration = fb[0].xpath("./duration")[0]
    bassnote = deepcopy(bass_child)
    bassnote.xpath("./duration")[0].text = new_duration.text

    # remove <dot> if exists since that alters note and we're making a new one entirely
    for dot in bassnote.xpath("./dot"):
        dot.getparent().remove(dot)

    # old note / divisions to get "standard value". then multiply by fb duration / old duration to split further between the fb
    # which cancels out to fb duration / divisions
    # NOTE: may not handle tuplets very well. also assumes the <divisions> value is the same across bass and continuo.
    new_note_value = round(int(new_duration.text) / divisions, 2)
    new_note_type = standardNoteTypeValue.get(new_note_value)
    dot_num = 0
    if (new_note_type is None):
        # creates dotted note
        # print("Error: no note type found for note value of", new_note_value, ". Defaulting to crotchets")
        # new_note_type = "quarter"
        # print(new_note_value)
        (new_note_type, dot_num) = create_dotted_note(new_note_value)

    type_xml=bassnote.xpath("./type")[0]
    type_xml.text = new_note_type
    # print(etree.tostring(bassnote, encoding="unicode", pretty_print=True))

    # add dots if they're there
    for i in range(dot_num):
        type_xml.addnext(etree.XML("<dot/>"))

    new_duration.getparent().remove(new_duration)
    fb_bass = append_lyrics_to_bass(bassnote, fb)
    return fb_bass


# goes up to double dots max
# returns (base note type, number of dots to add)
def create_dotted_note(note_value):
    lower_bound = 32
    upper_bound = 0.03

    note_value_list = list(standardNoteTypeValue)
    # upper bound starts at bottom, goes up until gets to first value that is more than than note value
    counter = 1
    while (upper_bound < note_value and counter < len(note_value_list)):
        upper_bound = note_value_list[counter]
        counter +=1

    # lower bound starts at the top, goes down until gets to first value that is less than the note value
    counter = len(note_value_list) - 2
    while (lower_bound > note_value and counter > 0):
        lower_bound = note_value_list[counter]
        counter -=1


    # halve, see if it fits note value. if so return dots = 1 and lower bound note
    # if it doesn't, halve again between that bound and upper bound. 
    base_note_type = standardNoteTypeValue.get(lower_bound)
    
    middle = lower_bound + round((upper_bound-lower_bound)*0.5, 2)
    if (middle == round(note_value, 2)):
        return (base_note_type, 1)
    
    middle = middle + round((upper_bound- middle)*0.5, 2)
    if (middle == round(note_value, 2)):
        return (base_note_type, 2)
    
    print("Could not find dotted equivalent for", note_value, ". Defaulting to two dots. ")
    return (base_note_type, 2)



# transforms into lyrics xml tag
# if multiple have to add number attribute
# if has duration element keep that but remove later ADD TO FIRST LYRIC ONLY (of numbered elements)
def turn_FBxml_into_lyrics(FBxml: Element, continued_FB = None) -> []:
    lyrics = []

    figures = FBxml.xpath("./figure")
    duration = FBxml.find("duration")
    for i in range (len(figures)):
        figure:Element = figures[i]
        number = i+1
        lyric = etree.Element("lyric", number =str(number) )

        # appends duration to first <lyric> for use in later processing
        if (i == 0 and duration is not None):
            dur = etree.SubElement(lyric, "duration")
            dur.text = duration.text
        
        # add <figure-number> and modifier. 
        # NOTE: if has modifier and no number, assumed to be 3 by music21.
        # https://github.com/cuthbertLab/music21/blob/master/music21/figuredBass/notation.py read this for accepted notation
        fig_num = figure.findtext("figure-number")
        prefix = figure.findtext("prefix")
        suffix = figure.findtext("suffix")

        extend = figure.xpath("./extend")
        extend_type = None
        if (len(extend) > 0):
            extend:Element = extend[0]
            extend_type = extend.get("type")
            extend_type = extend_type.lower() if extend_type is not None else None


        fig_string = ""
        # turn prefix and suffix into equivalent m21 notations
        # NOTE: figured bass module assumes suffixed modifiers are a new note - hence put in front instead 
        if prefix is not None:
            prefix_m21 = XMLToFBModifiers.get(prefix)
            text = prefix_m21 if prefix_m21 is not None else prefix

            fig_string = fig_string + text

        if suffix is not None:
            suffix_m21 = XMLToFBModifiers.get(suffix)
            text = suffix_m21 if suffix_m21 is not None else suffix

            fig_string = fig_string + text

        if fig_num is not None:
            fig_string = fig_string + fig_num
        

        fig_string_xml = etree.SubElement(lyric, "text")
        fig_string_xml.text = fig_string.strip()


        # if has extend tag need to process it, treat as adding another FB under the type = stop or type = continue for lyrics
        if extend_type == "start":
            continued_FB = fig_string.strip()
        elif extend_type =="continue" and continued_FB is not None:
            fig_string_xml.text = continued_FB
        elif extend_type=="stop" and continued_FB is not None:
            fig_string_xml.text = continued_FB
            continued_FB = None
        
        lyrics.append(lyric)

    return (lyrics, continued_FB)

        
    # check if multiple <figure>s - these are converted to 1:1 <lyrics> with number appended.
        #convert.
        # check if duration element is there.  duration added to first one as subelement if so
    # no multiple <figure>
        #convert.
        # check duration element add if so



def main():
    # score_path = "./chorales/FB_source/musicXML_master/BWV_177.05b_FB.musicxml"
    score_path = "./chorales/FB_source/musicXML_master/BWV_248.28_FB.musicxml"
    # score_path = "./temp/test.musicxml"

    fb = extract_FB(score_path, use_music21_realisation = True, return_whole_scoretree=True)

    file = open("temp/test_fb.musicxml", "wb")
    file.write(etree.tostring(fb, pretty_print=True))
    file.close()
    # print(etree.tostring(fb, encoding="unicode", pretty_print=True))

if __name__ == "__main__":
    main()