import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for i in range(0, len(test_set.get_all_Xlengths())):
    	feature_lists_sequences, sequences_length = test_set.get_item_Xlengths(i)
    	logL = {}

    	for word, model in models.items():
	    	try:
	    		score = model.score(feature_lists_sequences, sequences_length)
	    		logL[word] = score
	    	except:
	    		logL[word] = float("-inf")

    	probabilities.append(logL)

    	guess = max(logL, key = logL.get)
    	guesses.append(guess)

    return probabilities, guesses
