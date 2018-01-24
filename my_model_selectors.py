import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on BIC scores

        score_best = float('inf')
        model_best = None
        
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                n_features = self.X.shape[1]
                n_params = n_components * (n_components - 1) + 2 * n_features * n_components
                logN = np.log(self.X.shape[0])
                bic = -2 * logL + n_params * logN
                if bic < score_best:
                    score_best = bic
                    model_best = model
            except:
                pass
    
        model_best = model_best if model_best is not None else self.base_model(self.n_constant)
        return model_best


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        model_best = None
        score_best = float("-inf")

        words_other = list(self.words)
        words_other.remove(self.this_word)

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)

                score_sum = 0.0

                for word in words_other:
                    X, lengths = self.hwords[word]
                    score_sum += model.score(X, lengths)

                dic = logL - (score_sum / (len(self.words) - 1))

                if dic > score_best:
                    score_best = dic
                    model_best = model
            except:
                pass
        
        model_best = model_best if model_best is not None else self.base_model(self.n_constant)
        return model_best


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def cv_score(self, n):
        scores = []
        split_method = KFold(n_splits=2)

        for train_idx, test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(train_idx, self.sequences)

            model = self.base_model(n)
            X, l = combine_sequences(test_idx, self.sequences)

            scores.append(model.score(X, l))
        return np.mean(scores), model


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection using CV
        try:
            best_score = float("-Inf")
            best_model = None
            for n in range(self.min_n_components, self.max_n_components+1):
                score, model = self.cv_score(n)
                if score > best_score:
                    best_score = score
                    best_model = model
            return best_model
        except:
            return self.base_model(self.n_constant)
