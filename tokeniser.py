"""
Class to create a tokeniser object, that encodes figured bass notations to numbers. 
Used during the machine learning pipeline.
"""
class Tokeniser:
    tokens: dict[str, int] = {

    }

    # for converting from muspy to music21 format
    # value has commas inbetween
    tokens_commas: dict[int, str] = {

    }
    
    # start token should be 1 after the first empty number after pitch midi
    # 129 is used as a start of sentence token
    def __init__(self, start_token = 130, max_token = 280) -> None:
        self.start_token = start_token
        self.max_token = max_token

        # add None token
        self.tokens["None"] = start_token
        #add Unknown token since there's always possibility of unknown tokens showing up
        self.tokens["Unknown"] = start_token + 1

        # next free number to assign an FB to
        self.next = start_token + 2

    def get(self, fb_string):
        return self.tokens.get(fb_string, self.tokens.get("Unknown"))
    
    def get_with_commas(self, token):
        return self.tokens_commas.get(token, "None")
    
    # NOTE: could refactor this so only fb_separate needed
    def add(self, fb_string, fb_separate):
        token = self.tokens.get(fb_string)
        if (len(fb_string) == 0):
            print("Empty fb string - assign None")
            return self.tokens.get("None")

        if token is None and self.next > self.max_token:
            print("Maximum token number reached - encoding as Unknown")
            return self.tokens.get("Unknown")
        elif token is None:
            self.tokens[fb_string] = self.next
            self.next +=1

            new_token = self.tokens.get(fb_string)
            # add the comma'd version of the fb_string
            self.tokens_commas[new_token] = ",".join(fb_separate)

            return new_token
        # if token already exists
        else:
            return token
        
    
    def save(self):
        return {
            "tokens": self.tokens,
            "start": self.start_token,
            "max": self.max_token,
            "next": self.next,
            "w_commas": self.tokens_commas
            } 
    
    def get_none(self):
        return self.tokens.get("None")
    

    def get_max_token(self):
        length = len(self.tokens)

        # since single start token is like index 0 of a 1-length array, hence -1 
        return self.start_token + length - 1
    

    def load(self, state):
        self.start_token = state.get("start")
        self.next = state.get("next")
        self.max_token = state.get("max")

        self.tokens = state.get("tokens")
        self.tokens_commas = state.get("w_commas")

    # returns a val-key dictionary
    # will work since vals are unique too 
    def get_reversed_dict(self):
        reversed = dict((val, key) for key, val in self.tokens.items())
        return reversed

    def get_key(self,val) -> str:
        reversed = self.get_reversed_dict()
        key = reversed[val]

        return key
