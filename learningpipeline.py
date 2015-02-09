import readnarratives
import numpy




def getModelData(data):
	X_text = [narrative[0] for narrative in data]
	y = numpy.array([narrative[1] for narrative in data])

	return X_text,y

def processText(text_data):
	clean_text = []
	for text in text_data:
		clean_text.append(removeParenthesesText(text))
	return clean_text
def removeParenthesesText(text):
	leftp = text.find('(')
	rightp = text.find(')')

	if leftp>0 and rightp>0:
		new_text = text[:leftp] + text[rightp+1:]
		print(new_text)
		return removeParenthesesText(new_text)
	else: 
		return text 

def main():
	data = readnarratives.loadNarrativeData('agency')
	X,y = getModelData(data)

	print(processText(X))

	#print(removeParenthesesText('ajhflaskdjfh adjfhalskfj (adjfalskfjhas) adfhaljskdfh (ajsdhflkasdf)'))
if __name__ == '__main__':
	main()