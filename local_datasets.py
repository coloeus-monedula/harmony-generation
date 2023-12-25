from typing import Any
import muspy.datasets as datasets
from muspy import DatasetInfo, Lyric, read_musicxml
from muspy.music import Music
from torch.utils.data import Dataset
import torch



class MuspyChoralesDataset(datasets.FolderDataset):
    _info = DatasetInfo(license="Creative Commons Attribution 4.0 International", name="Bach Chorales Figured Bass (BCFB)", description="The complete 139 Johann Sebastian Bach chorales with figured bass encodings in MusicXML, **kern, and MEI formats, based on the Neue Bach Ausgabe (NBA) critical edition", homepage="10.5281/zenodo.5084913")
    _extension = "musicxml"
    resolution = 0

    def __init__(self, root, resolution, convert=False, kind='json', n_jobs=1, ignore_exceptions=True, use_converted=None ):
        
        super().__init__(root, convert=convert, kind=kind, n_jobs=n_jobs, ignore_exceptions=ignore_exceptions, use_converted=use_converted )
        self.resolution = resolution


    # # should return the i-th data sample as a muspy.Music object
    def __getitem__(self, index) -> Music:
        return super().__getitem__(index)

    # __len__ should return the size of the dataset
    def __len__(self) -> int:
        return super().__len__()
    
    # function based off __getitem__ code
    def get_by_filename(self, filename) -> Music:
        # should get single pointer back
        filtered = [f for f in self._filenames if filename in f.__str__()]

        if (len(filtered) == 0):
            raise FileNotFoundError

        if (len(filtered) > 1):
            print("Multiple items with filename found - picking first one.")
        
        return self._factory(filtered[0])


    def read(self, filename: Any) -> Music:
        return read_musicxml(filename, resolution=self.resolution)
    

    # NOTE: BELOW FUNCTION DOESN'T WORK - KEPT FOR POSTERITY.
    # adds the rest of the figured bass lyrics to the bassline
    # can't rely on muspy's lyrics - seems to be a bug in there that sometimes causes duplication so we have to do it ourselves
    # may as well do it in the tokenisation process 
    def complete_FB_lyrics(self, music21_lyrics: zip, filename):
        muspy_obj = self.get_by_filename(filename)
        # get muspy resolution - shows how many timesteps per quarter note. .quarterLength for m21 objects show how many quarter note lengths the Note is. do float(note * resolution) to get timesteps
        print(muspy_obj.resolution)
        muspy_lyrics = muspy_obj.tracks[-1].lyrics

        muspy_i = 0
        print(len(muspy_lyrics))
        filtered = [tup for tup in music21_lyrics if tup[0] is not None]
        print(len(filtered))

        if (len(filtered) != len(muspy_lyrics)):
            raise Exception("Muspy and Music21 lyric list are not equal length")

        for el in filtered:
            first, *others = el
            # print(first, others)

            single_lyric: Lyric = muspy_lyrics[muspy_i]
            lyric_str: str = single_lyric.lyric.strip()

            if (first !=lyric_str):
                print("Warning: lyrics are not equal value", first, lyric_str)

            for val in others:
                if val is not None:
                    lyric_str+=val.text.strip()

            single_lyric.lyric = lyric_str

            muspy_i+=1

   
#    where x and y are split
class PytorchSplitChoralesDataset(Dataset):
    def __init__(self, pt_file) -> None:
        self.chorales_dict: dict[str, (torch.Tensor, torch.Tensor)] = torch.load(pt_file)

    def __len__(self) -> int:
        return len(self.chorales_dict)
    
    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        chorales_list = list(self.chorales_dict.values())
        return chorales_list[index]
    
    def getname (self, index) -> str:
        chorales_list = list(self.chorales_dict.keys())
        return chorales_list[index]

    def get_by_filename(self, name) -> (torch.Tensor, torch.Tensor):
        tensor_tuple = self.chorales_dict.get(name)
        if tensor_tuple is None:
            raise ValueError("Tensor for ", name, "not found")
        
        return tensor_tuple


class PytorchChoralesDataset(Dataset):
    def __init__(self, pt_file):
        self.chorales_dict: dict[str, torch.Tensor] = torch.load(pt_file)


    def __len__(self):
        return len(self.chorales_dict)
    

    def __getitem__(self, index) -> torch.Tensor:
        chorales_list = list(self.chorales_dict.values())
        return chorales_list[index]
    

    def get_by_filename(self, name) -> torch.Tensor:
        tensor = self.chorales_dict.get(name)
        if tensor is None:
            raise ValueError("Tensor for ", name, "not found")
        
        return tensor


