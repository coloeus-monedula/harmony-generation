import glob
from os import path, makedirs
from extract_baseline import extract_FB
from lxml import etree
import muspy
from music21 import converter

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
            print("file: ", f)
            added_FB = extract_FB(f, use_music21_realisation=True, return_whole_scoretree=True)

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



def tokenise_FB():
    pass


# folder is path to converted FB xml
def add_tokenised_FB_dataset(converted_folder):
    folder_glob = path.join(converted_folder, "*.musicxml")
    files = glob.glob(folder_glob)


    for f in files:
        # read into muspy as musicxml USING ORIGINAL SOCRES
        # to avoid bass splitting workaround that was needed for music21
        # turn all into pytorch dataset - add identifier tag in class

        # read lyrics in from music21 using converted folder
        # tokenise FB
        # piece together
        pass


    pass




def main():
    in_folder = "chorales/FB_source/musicXML_master"
    out_folder = "added_FB"
    add_FB_to_scores(in_folder, out_folder, verbose=True)



if __name__ == "__main__":
    main()