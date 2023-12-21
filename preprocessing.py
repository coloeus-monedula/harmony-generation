
from os import path, makedirs
import dill as pickle

from muspy import Music
from extract_baseline import extract_FB
from lxml import etree
from music21 import converter, stream, chord, note as m21_note
from local_datasets import MuspyChoralesDataset, PytorchChoralesDataset as Chorales, PytorchSplitChoralesDataset as SplitChorales
from torch.utils.data import DataLoader, TensorDataset
import glob, torch, shutil, numpy as np, random
import requests, zipfile, io, re
from tokeniser import Tokeniser

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


def add_FB_to_scores(in_folder, filtered_folder, out_folder, verbose):
    folder_glob = path.join(in_folder, "*.musicxml")
    files = glob.glob(folder_glob)
    invalid_num = 0

    if not path.exists(out_folder):
        makedirs(out_folder)

    if not path.exists(filtered_folder):
        makedirs(filtered_folder)


    for f in files:
        if check_score_format(f, verbose) == True:
            # save to filtered folder
            basename = path.basename(f)
            filtered = path.join(filtered_folder, basename)
            shutil.copy(f, filtered)

            added_FB = extract_FB(f, use_music21_realisation=True, return_whole_scoretree=True, remove_OG_accomp=True)

            # save added FB
            added = path.join(out_folder, basename)
            with open(added, "wb") as file:
                file.write(etree.tostring(added_FB, pretty_print=True))

        else:
            if verbose:
                print("Ruled out: ", f)
                invalid_num+=1
    
    if verbose:
        print(invalid_num,"/", len(files),"files were invalid")



# folder is path to converted FB xml
# resolution = how many notes per crotchet - goes up to hemisemiquavers by default
def create_pytorch_train_dataset(filtered_folder, torch_file, resolution, split):

    # TODO: change this to original scores since we don't need to read lyrics into muspy obj anymore
    # TODO: though with how the dataset is encoded in numbers it encodes pitch but not duration of a single note, so does it matter?
    chorales = MuspyChoralesDataset(filtered_folder, resolution)

    dataset = chorales.to_pytorch_dataset(factory=FB_and_pianoroll_factory)

    dataset_dict = make_pytorch_dict(chorales, split, dataset)
    # print(dataset.__class__)
    # https://stackoverflow.com/questions/68617340/pytorch-best-practice-to-save-big-list-of-tensors or save tensors individually?
    # for 
    torch.save(dataset_dict, torch_file)
    
    return chorales


# TODO: given path to test dataset and path to existing tokens, loads each as mmuspy Music Objects/Musp
# runs it through FB and pianoroll
# save as dict for 
def create_pytorch_test_dataset(test_folder, token_file, torch_file, m21_lyrics_folder, resolution, split):

    chorales = MuspyChoralesDataset(test_folder, resolution)
    with open(token_file, "rb") as f:
        token_data = pickle.load(f)

    tokens = Tokeniser()
    tokens.load(token_data)

    dataset = []
    for i in range(len(chorales)):
        score = chorales[i]
        pytorch_score = FB_and_pianoroll(score, tokens, m21_lyrics_folder, is_test=True)

        dataset.append(pytorch_score)

    dataset_dict = make_pytorch_dict(chorales, split, dataset)
    torch.save(dataset_dict, torch_file)

    return chorales


def make_pytorch_dict(chorales, split, dataset):
    dataset_dict: dict[str, TensorDataset] = {}

    for i in range(len(dataset)):
        filename = chorales[i].metadata.source_filename

        if split:
            # get the alto - bass voice part by itself
            (tensor_x_start, tensor_y, tensor_x_fin) = torch.tensor_split(dataset[i],  (1, 4), dim=1)
            tensor_x = torch.cat((tensor_x_start, tensor_x_fin), dim=1)

            # timestep+1 S, Acc, FB
            x_plus_1 = tensor_x.detach().clone()[1:]
            # remove last line (of 0s)
            tensor_x = tensor_x[:-1]
            tensor_y = tensor_y[:-1]
            # add timestep+1 info to y
            tensor_y = torch.cat((tensor_y, x_plus_1), dim=1)

            dataset_dict[filename] = (tensor_x, tensor_y)
        else:
            tensor = dataset[i]
            dataset_dict[filename] = tensor
    return dataset_dict


def tokenise_FB(lyrics: list[m21_note.Lyric], is_test:bool):
    fb_string = ""

    # this is for converting the generated score back to music21 format
    fb_separate = []
    for lyric in lyrics:
        stripped = lyric.text.strip()
        fb_string+=stripped
        if len(stripped) != 0:
            fb_separate.append(stripped)

    # so we don't contaminate the tokenisation
    if is_test:
        token = empty_tokens.get(fb_string)
        if (token == empty_tokens.tokens["Unknown"]):
            print("Token in testset is Unknown")
        return token
    else:
        return empty_tokens.add(fb_string, fb_separate)



# factory method to call, uses pianoroll conversion inside but also adds on encoded FB using tokenise_FB and ignores velocity
# NOTE: CAN'T USE MUSPY PIANOROLL FORMAT, make our own with pitch numbers instead - https://github.com/ageron/handson-ml3/blob/main/15_processing_sequences_using_rnns_and_cnns.ipynb
# for use when converting a Muspy dataset for training
def FB_and_pianoroll_factory(score: Music):
    return FB_and_pianoroll(score, empty_tokens, m21_lyrics_folder, is_test=False)

# https://salu133445.github.io/muspy/_modules/muspy/outputs/pianoroll.html#to_pianoroll_representation
# based off original pianoroll code
def FB_and_pianoroll(score:Music, tokens:Tokeniser, m21_lyrics_folder:str, is_test: bool ):
    filename = score.metadata.source_filename
    resolution = score.resolution

    # get figured bass since muspy doesn't read it in
    m21 = converter.parseFile(m21_lyrics_folder+"/"+filename)
    fb = m21.parts[-1]

    fb_length = int(fb.duration.quarterLength * resolution)
    # array specification follows pianoroll representation spec
    # length+1 so last bar is a bar of silence, so slicing t+1 for last index still works
    fb_array = np.full(fb_length+1, tokens.get_none(), np.int16)

    # last bar of silence, won't have default FB so make silence token
    fb_array[-1] = SILENCE
    fb_timestep = 0

    for el in fb.recurse().notes:
        # find equivalent in timesteps, following how muspy does it
        duration = int(el.duration.quarterLength * resolution)
        lyrics = el.lyrics
        if len(lyrics) != 0:
            token = tokenise_FB(lyrics, is_test)
            fb_array[fb_timestep: fb_timestep+duration] = token

        fb_timestep+=duration


    # convert the rest of the notes into pitches
    pianoroll = np.full((fb_length+1, len(score.tracks) + 1), SILENCE, np.int16)
    for track_num in range(len(score.tracks)):
        notes = score.tracks[track_num].notes
        for note in notes:
            pianoroll[note.time:note.end, track_num] = note.pitch


    # add FB
    pianoroll[:,-1] = fb_array

    torch_vers = torch.from_numpy(pianoroll)
    return torch_vers.long()



# given a filtered musicxml folder, selects n random files and moves to test/ folder
# can also manually move musicxml files
def move_test_chorales(dest, source, n):
    if not path.exists(dest):
        makedirs(dest)

    folder_glob = path.join(source, "*.musicxml")
    files = glob.glob(folder_glob)
    selected = random.sample(files, n)

    for file in selected:
        shutil.move(file, dest)


def get_chorales(url, dest_folder ):
    request = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(request.content))

    # make into expected chorales/ file path format
    for file in z.infolist():
        # matches the .musicxml that is immediately under musicXML and doesn't go into further folders due to "(?!.*/)"
        match = re.match("juyaolongpaul-Bach_chorale_FB-0873cc7/FB_source/musicXML_master/(?!.*/).*\.musicxml", file.filename)
        if match:
            print(file.filename)
            if file.is_dir():
                continue

            #remove dir info
            file.filename = path.basename(file.filename)

            z.extract(file, dest_folder)

    z.close()


# filename - lyrics object
m21_lyrics_folder = ""

num = 250
empty_tokens = Tokeniser(max_token = num)
SILENCE = 128

def main():
    in_folder = "./chorales/FB_source/musicXML_master"
    # original scores but without ineligible scores - use for muspy dataset
    filtered_folder = "filtered"
    out_folder = "added_FB"

    test_folder = "test_scores"
    torch_save = "artifacts/"+str(num)+"_preprocessed.pt"
    torch_test_save = "artifacts/"+str(num)+"_preprocessed_test.pt"
    token_save = "artifacts/"+str(num) +"_tokens.pkl"
    resolution = 8

    if not path.exists("chorales"):
        print("Downloading chorales.")
        get_chorales("https://zenodo.org/records/5084914/files/juyaolongpaul/Bach_chorale_FB-v2.0.zip?download=1", in_folder)


    add_FB_to_scores(in_folder, filtered_folder, out_folder, verbose=True)

    global m21_lyrics_folder
    m21_lyrics_folder = out_folder

    # split into train test dataset here - or, can manually move scores from a filtered folder to test folder
    # this is done before train dataset so tokenisation doesn't occur, but actual conversion done after train dataset
    # move_test_chorales(test_folder, filtered_folder, 3)

    create_pytorch_train_dataset(filtered_folder, torch_save,resolution, split=True)

    print(empty_tokens.tokens)
    with open(token_save, "wb") as f:
        pickle.dump(empty_tokens.save(), f)

    create_pytorch_test_dataset(test_folder, token_save, torch_test_save, m21_lyrics_folder, resolution, split=True)
    # file = "./chorales/FB_source/musicXML_master/BWV_248.59_FB.musicxml"
    # combine_bassvoice_accomp(file)


if __name__ == "__main__":
    main()