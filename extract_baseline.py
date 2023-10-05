from xml.etree.ElementTree import Element
from lxml import etree
from copy import deepcopy


# modifiers dictionary taken from 
# https://github.com/cuthbertLab/music21/blob/master/music21/figuredBass/notation.py 
modifiersDictXmlToM21 = {
    'sharp': '#',
    'flat': 'b',
    'natural': '\u266e',
    'double-sharp': '##',
    'flat-flat': 'bb',
    'backslash': '\\',
    'slash': '/',
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


def extract_FB(score_path, use_music21_realisation = False, return_whole_scoretree = False):
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
        divisions = int(parts[-2].xpath('measure[1]/attributes/divisions')[0].text)
        bass = list(parts[-2].iter("measure"))
        continuo_measures = list(continuo.iter("measure"))
        for i in range(len(bass)):
            measure = combine_bassvoice_and_FB(continuo_measures[i], bass[i], divisions)
            fb.append(measure)

        #add part-list info here
        fb_scorepart = etree.Element("score-part", id="FB")
        etree.SubElement(fb_scorepart, "part-name").text = "Bass and FB"

        part_list = root.xpath("./part-list")[0]
        part_list.append(fb_scorepart)

    else:
        for measure in continuo.iter("measure"):
            fb_only = create_FB_measure(measure)
            fb.append(fb_only)
    
    if (return_whole_scoretree):
        root.append(fb)
        # NOTE: scoretree returned if not doing music21 realisation is not "proper" MusicXML
        return score_tree
    else:
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

# NOTE: assumes bass and continuo have the same notes albeit transposed an octave.
def combine_bassvoice_and_FB(continuo: Element, bass: Element, divisions: int) -> Element:
    bass_attrib = dict(bass.attrib)

    FB_measure:Element = etree.Element("measure")

    for attrib in bass_attrib:
        FB_measure.set(attrib, bass_attrib.get(attrib))

    # traverse through bass and continuo part at same time
    continuo_children = continuo.xpath("./*")
    bass_children = bass.xpath("./*")
    bass_child: Element = bass_children[0]
    temp_fb = []

    # assuming bass and continuo have same notes continuo will always be equal or longer due to also having FB too 
    for child in continuo_children:
        if (child.tag == "figured-bass"):
            # 2D array
            temp_fb.append(turn_FBxml_into_lyrics(child))
        elif (child.tag == "note"):
            # advances bass to the next <note>. anything that isn't <note> gets appended to FB measure
            while (bass_child is not None and bass_child.tag != "note"):
                FB_measure.append(deepcopy(bass_child))
                bass_child = bass_child.getnext()

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
    # fetch duration value from first <lyric> and add to copied <note>
    # then remove it from <lyrics>

    new_duration = fb[0].xpath("./duration")[0]
    bassnote = deepcopy(bass_child)
    bassnote.xpath("./duration")[0].text = new_duration.text

    # old note / divisions to get "standard value". then multiply by fb duration / old duration to split further between the fb
    # which cancels out to fb duration / divisions
    # NOTE: may not handle tuplets very well. also assumes the <divisions> value is the same across bass and continuo.
    new_note_value = round(int(new_duration.text) / divisions, 2)
    new_note_type = standardNoteTypeValue.get(new_note_value)
    if (new_note_type is None):
        print("Error: no note type found for note value of" + new_note_value + ". Defaulting to crotchets")
        new_note_type = "quarter"
    
    bassnote.xpath("./type")[0].text = new_note_type

    fb[0].remove(new_duration)
    fb_bass = append_lyrics_to_bass(bassnote, fb)
    return fb_bass




# transforms into lyrics xml tag
# if multiple have to add number attribute
# if has duration element keep that but remove later ADD TO FIRST LYRIC ONLY (of numbered elements)
def turn_FBxml_into_lyrics(FBxml: Element) -> []:
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
        # TODO: confirm bach chorales doesn't use <extend>. also how to deal with backslash? just write it in and see what happens
        # https://github.com/cuthbertLab/music21/blob/master/music21/figuredBass/notation.py read this for accepted notation
        fig_num = figure.findtext("figure-number")
        prefix = figure.findtext("prefix")
        suffix = figure.findtext("suffix")

        fig_string = ""
        # turn prefix and suffix into equivalent m21 notations
        if prefix is not None:
            prefix_m21 = modifiersDictXmlToM21.get(prefix)
            text = prefix_m21 if prefix_m21 is not None else prefix

            fig_string = fig_string + text

        if fig_num is not None:
            fig_string = fig_string + fig_num
        
        if suffix is not None:
            suffix_m21 = modifiersDictXmlToM21.get(suffix)
            text = suffix_m21 if suffix_m21 is not None else suffix

            fig_string = fig_string + text

        fig_string_xml = etree.SubElement(lyric, "text")
        fig_string_xml.text = fig_string

        
        lyrics.append(lyric)

    return lyrics

        
    # check if multiple <figure>s - these are converted to 1:1 <lyrics> with number appended.
        #convert.
        # check if duration element is there.  duration added to first one as subelement if so
    # no multiple <figure>
        #convert.
        # check duration element add if so



def main():
    score_path = "chorales/FB_source/musicXML_master/BWV_3.06_FB.musicxml"
    fb = extract_FB(score_path)
    print(etree.tostring(fb, encoding="unicode", pretty_print=True))

if __name__ == "__main__":
    main()