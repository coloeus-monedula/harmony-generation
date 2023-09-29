from lxml import etree


def extract_FB(score_path):
    score_tree = etree.parse(score_path)
    root = score_tree.getroot()

    # get all parts
    parts = list(root.iter("part"))
    
    # fb is always contained in the last part
    bass = parts[-1]
    # print(etree.tostring(bass, encoding="unicode"))
    
    # how many divisions in a quaver note
    divisions = int(bass.xpath('measure[1]/attributes/divisions')[0].text)
    print(divisions)

    fb = etree.Element("part", id="FB")

    # measures =  bass.iter("measure")
    for measure in bass.iter("measure"):
        fb_only = create_FB_measure(measure)

    # etree.SubElement()

    # TODO: basically extract this into its own FB xmltree - preserve as much info as possible

# returns a new measure that contains solely FB information only
# https://stackoverflow.com/questions/4005975/etree-clone-node 
# TODO: make sure measure info is also saved
# TODO: do it so that you pick up everything EXCEPT NOTE
# def create_FB_measure(measure):


def main():
    score_path = "chorales/FB_source/musicXML_master/BWV_3.06_FB.musicxml"
    extract_FB(score_path)

if __name__ == "__main__":
    main()