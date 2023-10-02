# https://pypi.org/project/pipenv/
from music21 import *

# configure.run()

piece = converter.parseFile("chorales/FB_source/musicXML_master/BWV_3.06_FB.musicxml")
# piece = converter.parse("chorales/test.musicxml")
print(len(piece.parts))

piece.show('text')
# continuo = piece.parts[-1]
# continuo.show('text')
# measure1 = continuo.measure(1)
# measure1.show('text')

# gets all fb notation since they're attached to notes
# however some fb notation isn't necessarily attached TO any notes..
piece.flatten().lyrics()

# https://web.mit.edu/music21/doc/moduleReference/moduleFiguredBassRealizer.html#functions easiest way to get figured bass??
# not working

# may need to manually extract via xml parser

# https://web.mit.edu/music21/doc/moduleReference/moduleFiguredBassExamples.html 
# fb = figuredBass.realizer.figuredBassFromStream(continuo)
# fb.generateBassLine().show()
# # fb.addElement(note.Note('C#3'), '6')
# fbRules = figuredBass.rules.Rules()
# fbRealization = fb.realize(fbRules)
# fbRealization.generateRandomRealizations().show('text')


#music21 assumes figured bass are lyrics but they've put it as figured-bass
# fbLine = figuredBass.realizer.FiguredBassLine(key.Key('B'), meter.TimeSignature('3/4'))
# fbLine.addElement(note.Note('B2'))
# fbLine.addElement(note.Note('C#3'), '6')
# fbLine.addElement(note.Note('D#3'), '6')
# fbLine.generateBassLine().show('text')

# music21 does have a figuredBass as well as figuredBass rules 
# figuredBassFromStream if you convert FB to lyrics could work but lyrics seem to have to be connected to a note?
# 


# figured bass notation is stored seperately?? maybe the baseline needs to be bespoke programmed?

