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
(makexmlnarratives.py)
The narratives, in their original form, were word documents. We've turned these into xml documents.

The idea with the xml documents are that they are as close to the original narraitve that a human coder would code. We do give the machine a little help, though. We use an XML scaffolding so that we can tag different parts of the narrative and have the scene, the score, and any tagged examples. 

	The xml file format can be seen in samplexmlformat.xml

TAGGING SECTIONS OF THE XML FILES
When we tag sections of the document, we use the following format. 
	Note: Every part of the respondents answers are tagged. We don't have any text that exists outside of a tag. 

	<example dimension = "agency" score = "1.5">
	</example>

PROCESSING THE TEXT
(readnarratives.py, preprocessingtext.py)
This happens in two phases, reading the xml files, and then processing the text. 

First, we read the narrative xml files. We do this in a few parts, getting the xml object from reading, and then pulling out the scene and example texts with scores for each narrative. All of the data has the format
	[((scene text, scene score),[(ex text, ex score)])]

Next, we process the text, getting it ready for learning. In this section, we 
Remove anything in parentheses (these are mostly comments like "laughing" from the transcribers)

Then, we remove all characters that aren't alphabetical characters or spaces

Next we remove extra spaces, so that there is never more than one space between words. 

Then we get rid of any empty strings whose content has been removed by the processing. We also throw out the corresponding scores. 


GETTING DATA FOR LEARNING
(scorestenencesdata.py)

Here, we get the response from a narrative, and convert it into a list of scored sentences.

First, we get the narrative data from a pickle, Then, we split the data apart by scene.

Next, we take the scene text and split it apart by sentences. 

Then, we score each sentence based on which example it overlaps the most with. 



