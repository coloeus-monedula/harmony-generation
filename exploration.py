# https://pypi.org/project/pipenv/
from music21 import *

# configure.run()

piece = converter.parseFile("chorales/FB_source/musicXML_master/BWV_3.06_FB.musicxml")
print(len(piece.parts))

# piece.show()
continuo = piece.parts[-1]
# continuo.show('text')
measure1 = continuo.measure(1)
# measure1.show('text')


# https://web.mit.edu/music21/doc/moduleReference/moduleFiguredBassRealizer.html#functions easiest way to get figured bass??
# not working

# may need to manually extract via xml parser
fb = figuredBass.realizer.figuredBassFromStream(continuo)
fbRules = figuredBass.rules.Rules()
fbRealization = fb.realize(fbRules)
fbRealization.getNumSolutions()
fbRealization.generateAllRealizations().show('text')

