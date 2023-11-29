


class Tokeniser:
    tokens: dict[str, int] = {

    }
    
    # start token should be 1 after the first empty number after pitch midi
    # 129 is used as a start of sentence token
    def __init__(self, start_token = 130, max_token = 280) -> None:
        self.start_token = start_token
        self.max_token = max_token

        # add None token
        self.tokens["None"] = start_token

        # next free number to assign an FB to
        self.next = start_token + 1

    def get(self, fb_string):
        return self.tokens.get(fb_string, self.tokens.get("Unknown"))

    def add(self, fb_string):
        token = self.tokens.get(fb_string)

        # unknown already added to dict
        if token is None and self.next > self.max_token:
            print("Maximum token number reached - encoding as Unknown")
            return self.tokens.get("Unknown")
        elif token is None and self.next == self.max_token:
            print("Maximum token number reached - encoding as Unknown (First time)")
            self.tokens["Unknown"] = self.next
            self.next +=1

            return self.tokens.get("Unknown")

        elif token is None:
            self.tokens[fb_string] = self.next
            self.next +=1

            return self.tokens.get(fb_string)
        
        else:
            return token
        
    
    def save(self):
        return {
            "tokens": self.tokens,
            "start": self.start_token,
            "max": self.max_token,
            "next": self.next} 
    
    def get_none(self):
        return self.tokens.get("None")
    

    def get_max_token(self):
        length = len(self.tokens)

        # since start token is like index 0 of a 1-length array, hence -1 
        return self.start_token + length - 1
    

    def load(self, state):
        self.start_token = state.get("start")
        self.next = state.get("next")
        self.max_token = state.get("max")

        self.tokens = state.get("tokens")


# TODO: make into class, set a "start token" 
# if get beyond max_token , set as the unknown token
# input size then becomes max_token + 1 in RNN bc we also use 0

# tokeniser symbols from this: https://robertkelleyphd.com/home/FiguredBass1.pdf#page=2
# TODO: also look at how paper did their FB as it will differ slightly from the "standard" - more verbose for instance in 75
# decouple multiple figures, treat as independent - let model potentially figure it out, same with equivalent FB notations
tokeniser= {
    "None":150,
    "Unknown": 250,
    "#": 151,
    "b": 152,
    "n": 153,
    "2": 154, 
    "#2":155,
    "b2":156,
    "+2":157,
    "3": 158,
    "4": 159,
    "42":160,
    "+42": 161,
    "#42": 162,
    "43": 163,
    "5": 164,
    "+5":165,
    "#5":166,
    "b5":167,
    "53":168,
    "54": 169,
    "6":170,
    "#6":171,
    "63":172,
    "64":173,
    "642":174,
    "643":175,
    "65":176,
    "653":177,
    "7": 178,
    "742":179,
    "8":180,
    "9":181,
    "10":182,
    "7#52":183,
    "5#":184,
    "52": 185,
    "7#5":186,
    "n5": 187,
    "6+42": 188,
    "7n": 189,
    "6n":190,
    "7#":191,
    "74b2":192,
    "86": 193,
    "75":194,
    "n56":195,
    "n75":196,
    "6n5": 197,
    "6#5": 198,
    "6b5": 199,
    "n6": 200,
    "b6": 201,
    "#643": 202,
    "n642": 203,
    "n7": 204,
    "84": 205,
    "73": 206,
    "n643": 207,
    "864": 208,
    "6+4": 209,
    "6#4": 210,
    "6n42":211,
    "n6": 212,
    "5n":213,
    "96":214,
    "65n":215,
    "65#":216,
    "65b":217,
    "b75": 218,
    "7n5": 219,
    "8#":220,
}