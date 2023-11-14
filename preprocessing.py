import glob
from os import path, makedirs
from extract_baseline import extract_FB
from lxml import etree
from music21 import converter, stream, chord, musicxml
from local_datasets import ChoralesDataset
from torch.utils.data import DataLoader, TensorDataset

# music21 combines the two parts using chordify, takes the top layer of notes if there's a chord
# then export to musicxml, open and grab the part in xml, replace where bass is in original score
# output as musicxml or as file
def combine_bassvoice_accomp(file, return_type = "tree"):

    score = converter.parseFile(file)
    parts = score.parts

    b = parts[3]
    accomp = parts[4]

    new_score = stream.Score()
    new_score.append(b)
    new_score.append(accomp)

    chords = new_score.chordify()

    # single_notes = stream.Part()
    for c in chords.recurse().getElementsByClass(chord.Chord):
        c.sortAscending(inPlace=True)
        notes = c.notes
        # keep highest note as this preserves as much bass voice as possible
        for i in range(len(notes) -1):
            to_remove = notes[i]
            c.remove(to_remove)
    

    filepath = path.join("temp", path.basename(file))
    chords.write("musicxml", fp = filepath)

    combined_xml = etree.parse(filepath)
    combined_part = combined_xml.xpath("./part")[0]

    original = etree.parse(file)

    parts =original.xpath("./part")
    og_bass = parts[3]
    og_bass.getparent().replace(og_bass, combined_part)

    fb_scorepart = combined_xml.xpath("./part-list")[0][0]

    b_part_list = original.xpath("./part-list")[0][3]
    b_part_list.getparent().replace(b_part_list, fb_scorepart)
    
    # if return_type == "file":
    file = open("temp/test.musicxml", "wb")
    file.write(etree.tostring(original, pretty_print=True))
    file.close()

    # chords.show()
    # pass

# limiting the scores used to SATB + continuo bassline only
# given a music21 score, checks that there are 5 parts and that four of them is voice
def check_score_format(file, verbose, format = "satb"):
    score = converter.parseFile(file)


    if (format == "satb" and len(score.parts) != 5):
        if (verbose):
            print("Parts are ", len(score.parts))
        return False
    
    parts = score.parts
    voice_parts = 0
    for part in parts:
        if part.partName == "Voice":
            voice_parts+=1

    if (format == "satb" and voice_parts !=4):
        if (verbose):
            print("Number of voice parts: ", voice_parts)
            return False

    return True


def add_FB_to_scores(in_folder, out_folder, verbose):
    folder_glob = path.join(in_folder, "*.musicxml")
    files = glob.glob(folder_glob)
    invalid_num = 0

    if not path.exists(out_folder):
        makedirs(out_folder)

    for f in files:
        if check_score_format(f, verbose) == True:
            # print("file: ", f)
            added_FB = extract_FB(f, use_music21_realisation=True, return_whole_scoretree=True, remove_OG_accomp=True)

            basename = path.basename(f)
            filename = path.join(out_folder, basename)

            file = open(filename, "wb")
            file.write(etree.tostring(added_FB, pretty_print=True))
            file.close()
        else:
            if verbose:
                print("Ruled out: ", f)
                invalid_num+=1
    
    if verbose:
        print(invalid_num,"/", len(files),"files were invalid")



# folder is path to converted FB xml
def add_FB_to_muspy_dataset(converted_folder):
    folder_glob = path.join(converted_folder, "*.musicxml")
    files = glob.glob(folder_glob)

    chorales = ChoralesDataset(converted_folder)
    # print(first.tracks[-1].lyrics)

    # dataset = chorales.to_pytorch_dataset(representation="pianoroll")
    # then put into dataloader?

    for f in files:
        music21_obj = converter.parseFile(f)
        lyrics = music21_obj.parts[-1].lyrics()

        if lyrics == None:
            raise ValueError("FB part has no lyrics")

        # NOTE: if zip can't do None objects may show up with error if any of the keys are None
        print(f) #pass along f as filename
        key_num = len(lyrics.keys())
        if key_num == 3:
            zipped = zip(lyrics.get(1), lyrics.get(2), lyrics.get(3))
        elif key_num == 2:
            zipped = zip(lyrics.get(1), lyrics.get(2))
        elif key_num == 1:
            zipped = zip(lyrics.get(1))
        else:
            print ("Error: lyric keys number is ", key_num, ", defaulting to zipping 1")
            zipped = zip(lyrics.get(1))

        
        chorales.complete_FB_lyrics(zipped, f)

    return chorales

def tokenise_FB():
    tokeniser: {
        None:150,

    }

    pass


# factory method to call, uses pianoroll conversion inside but also adds on encoded FB using tokenise_FB
def FB_and_pianoroll():
    pass


# TODO: save both the pre-pytorch dataset and the post pytorch painorollls

def convert_to_pytorch_dataset(chorales: ChoralesDataset, pytorch_folder):
    pass


def main():
    in_folder = "chorales/FB_source/musicXML_master"
    out_folder = "added_FB"
    # add_FB_to_scores(in_folder, out_folder, verbose=True)

    add_FB_to_muspy_dataset(out_folder)

    # file = "./chorales/FB_source/musicXML_master/BWV_248.59_FB.musicxml"
    # combine_bassvoice_accomp(file)


if __name__ == "__main__":
    main()