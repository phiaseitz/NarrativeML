import readnarratives
import numpy



def getModelData(data):
	X_text = [narrative[0] for narrative in data]
	y = numpy.array([narrative[1] for narrative in data])

	return X_text,y


def main():
	data = readnarratives.loadNarrativeData('agency')
	X,y = getModelData(data)
	print (y)
if __name__ == '__main__':
	main()