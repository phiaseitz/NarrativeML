OVERVIEW
This is a project for coding narratives for themes using ML. This semester, I've been working on coding agency in the high low and turning point of life story narratives. 

This process has a few steps from start to finish
1.) Take the narratives that are currently .txt files, and make them something a little more readable. We chose to use .xml files for this
2.) Make .xml files from all the narratives tagging who is speaking when 
3.) Currently, we're in the process of doing this, but then we tag all the respondent text as we would score just that particular part of the passage. Often this is on a sentence by sentence basis, but sometimes it's more than one sentence or a phrase
4.) Process the tagged narratives. This includes pulling out just what the respondent says, removing any punctuation, and multiple spaces, and anything that's not actually spoken. 
5.) We then score each sentence of the tagged data based on how it was tagged in the original narraitve.
6.) These scored sentences are what we use for learning. Currently we're using ordinal regression
7.) We haven't 100% gotten here yet, but after we get a good model working, we'll combine sentence scores for a passage and generate a passage score from those sentences. 

FROM TXT TO XML
The narratives, in their original form, were word documents. We've turned these into xml documents using the following functions

	makeScoresDict:
		reads the csv of all the saved scores, and makes a dictionary that goes from number of narrative to all the scores for a particular narrative
	readFile: 
		takes in the number of a narrative in the appropraite folder and returns the list of lines of the txt file
	cleanText:
		Takes in the list of lines in the txt file and gets rid of all the empty lines and lines that were just whitespace. Also takes out any new line characters in any of the strings
	removeSpeakerFromLine:
		Takes in a line and returns the line without the speaker tag, if there is one.  (All lines where the speaker changes have this denoted as SPEAKER:  , and we don't want to have this in our learning data)
	addTextToCurrentScene:
		Takes in the xml object of the narrative, and the text of a passage as well as the current speaker and appends the passage to the current scene in the xml object, tagging the speaker as current speaker and labeling it as a passage. A passage is all the consecutive things one speaker says before the speaker changes
	linesToXML:
		Takes in list of the lines, the number of the narrative and the scores for agency and communion for each of the narratives. then searches for who the speaker is if there is one. Then, cycles through all the lines, adding each line to an appropriate passage, and each passage to the appropraite scene. Returns the xml object that is the converted narrative
	saveXML:
		Takes in the xml object of the narrative and the number of the narrative and saves the narrative as the appropriate file in the folder

	The xml file format can be seen in samplexmlformat.xml
