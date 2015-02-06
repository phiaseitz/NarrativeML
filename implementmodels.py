import readnarratives
import numpy



def getModelData(data):
	X = [1,2,3,4,5]
	y = numpy.array([narrative[1] for narrative in data])
	print (y)

def main():
	data = readnarratives.loadNarrativeData('agency')
	X,y = getModelData(data)
	print (y)
if __name__ == '__main__':
	main()