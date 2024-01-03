from music21 import *
import argparse

"""
Manually sets the environment path for Linux-using computers so that Music21 can open scores in MuseScore.
"""

parser = argparse.ArgumentParser(description="Set env path for linux builds")
parser.add_argument("location", choices=["uni","home"])
args = parser.parse_args()

if (args.location == "uni"):
    #uni linux
    environment.set('musicxmlPath', "/home/sh318/Documents/y5/diss/musescore.AppImage")
else:
    #home linux
    environment.set('musicxmlPath', "/home/corv/Applications/MuseScore-4.1.1.232071203-x86_64_be05f6a5f07179df118e3f5b42953855.AppImage")