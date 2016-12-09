import random
import pandas as pd
import numpy as np

lambdas = [.75, .5]
forums = [1, 2]



def mmr():
	return ['mmr1', 'mmr2', 'mmr3', 'mmr4', 'mmr5']

def topComments():
	return ['scored1', 'scored2', 'scored3', 'scored4', 'scored5']

def getRandomComment():
	return 'Random'

trueAssignment = []
columnA = []
columnB = []
columnC = []

df = pd.DataFrame(columns = None)
truth = pd.DataFrame(columns = None)


count = 0
for forum in forums:
	for l in lambdas:
		for c in range(2):
			scoredComments = topComments() # Top 5 scored comments
			mmrComments = mmr() # output of MMR Comemnts
			cComment = getRandomComment() #get a random comment (Kim)
			randomX = (random.getrandbits(1))
			if randomX == 0:
				trueAssignment.append('A')
				columnA.append(mmrComments)
				columnB.append(scoredComments)
			if randomX == 1:
				trueAssignment.append('B')
				columnA.append(scoredComments)
				columnB.append(mmrComments)
			columnC.append(cComment)



df['A'] = columnA
df['B'] = columnB
df['C'] = columnC
df['MMR Comments'] = trueAssignment

df = df.iloc[np.random.permutation(len(df))] # Randomize the ordering of the row
truth = df['MMR Comments']
df = df.drop('MMR Comments', 1)

df.to_csv('experiment.csv')
truth.to_csv('truth.csv')



