import glob
from os import path, makedirs

from muspy import Music
from extract_baseline import extract_FB
from lxml import etree
from music21 import converter, stream, chord, note as m21_note
from local_datasets import ChoralesDataset
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
np.set_printoptions(threshold=np.inf)




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

# filename - lyrics object
m21_lyrics_folder = ""

# folder is path to converted FB xml
def add_FB_to_muspy_dataset(converted_folder):
    folder_glob = path.join(converted_folder, "*.musicxml")
    files = glob.glob(folder_glob)

    chorales = ChoralesDataset(converted_folder)
    # print(first.tracks[-1].lyrics)


    dataset = chorales.to_pytorch_dataset(factory=FB_and_pianoroll)
    print(chorales[0].metadata.source_filename)
    print(dataset[0])
    return chorales


# gets a list of Lyrics
def tokenise_FB(lyrics: list[m21_note.Lyric]):
    return 170


tokeniser= {
    "None":150,

}

# factory method to call, uses pianoroll conversion inside but also adds on encoded FB using tokenise_FB and ignores velocity

# TODO: READ MUSIC21 HERE -> GET LAST FB PART 
# TODO: get muspy resolution - shows how many timesteps per quarter note. .quarterLength for m21 objects show how many quarter note lengths the Note is. do float(note * resolution) to get timesteps
#  for all notes in part:
# get duration of note in muspy timesteps and see if there is FB under it
# if no FB: encode as "None" value for that duration of timesteps
# if there is, tokenise the lyrics, return value (with default being another val eg. 180)
# results in array of length timesteps
# then: zip() tracks and FB together (check they first have same length) - use the encode notes func to init array. position is s/a/t/b/accomp/fb and not note.pitch however, and value is pitch
# one "internal" array for every timestep 
# need to check fb timestep legnth is equal to the others
# NOTE: CAN'T USE MUSPY PIANOROLL FORMAT, make our own with pitch numbers instead - https://github.com/ageron/handson-ml3/blob/main/15_processing_sequences_using_rnns_and_cnns.ipynb


# for i in m21.parts[-1].recurse().notes:
    # print(i.duration.quarterLength)
    # print(i.lyrics)


# https://salu133445.github.io/muspy/_modules/muspy/outputs/pianoroll.html#to_pianoroll_representation
# based off original pianoroll code
def FB_and_pianoroll(score: Music):
    filename = score.metadata.source_filename
    resolution = score.resolution

    m21 = converter.parseFile(m21_lyrics_folder+"/"+filename)
    fb = m21.parts[-1]

    fb_length = int(fb.duration.quarterLength * resolution)
    # array specification follows pianoroll representation spec
    # NOTE: do we NEED length+1? is length +1 just so the last bit is a bar of silence?
    fb_array = np.full(fb_length+1, tokeniser.get("None"), np.uint8)

    # last bar of silence, won't have default FB so make 0
    fb_array[-1] = 0
    fb_timestep = 0

    for el in fb.recurse().notes:
        # find equivalent in timesteps
        duration = int(el.duration.quarterLength * resolution)
        lyrics = el.lyrics
        if len(lyrics) != 0:
            token = tokenise_FB(lyrics)
            fb_array[fb_timestep: fb_timestep+duration] = token

        fb_timestep+=duration


    # convert the rest of the notes into pitches
    pianoroll = np.zeros((fb_length+1, len(score.tracks) + 1), np.uint8)
    for track_num in range(len(score.tracks)):
        notes = score.tracks[track_num].notes
        for note in notes:
            pianoroll[note.time:note.end, track_num] = note.pitch


    # add FB
    pianoroll[:,-1] = fb_array
    return pianoroll



# TODO: save both the pre-pytorch dataset and the post pytorch painorollls

def convert_to_pytorch_dataset(chorales: ChoralesDataset, pytorch_folder):
    pass


def main():
    in_folder = "chorales/FB_source/musicXML_master"
    out_folder = "added_FB"
    # add_FB_to_scores(in_folder, out_folder, verbose=True)
    global m21_lyrics_folder
    m21_lyrics_folder = out_folder

    add_FB_to_muspy_dataset(out_folder)

    # file = "./chorales/FB_source/musicXML_master/BWV_248.59_FB.musicxml"
    # combine_bassvoice_accomp(file)


if __name__ == "__main__":
    main()