import nltk, pickle, csv
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        '''
        Return the label that the majority of chosen
        classifiers choose for the input features
        '''
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        '''
        Return the confidence of the label chosen by
        chosen classifiers
        '''
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# Load the saved featureset
word_features_f = open("pickled_algos/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

def find_features(cleaned_script):
    '''
    Generates a featureset showing if words in word_features
    can be found in the input text 
    '''
    features = {}
    for w in word_features:
        features[w] = (w in cleaned_script)
    
    return features


# Open the trained algorithms 
open_file = open("pickled_algos/NaiveBayes_classifier.pickle", "rb")
NaiveBayes_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/SGDC_classifier.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()


# Create a new VoteClassifier with 5 classifiers
voted_classifier = VoteClassifier(NaiveBayes_classifier, LinearSVC_classifier, MNB_classifier,
                                  BernoulliNB_classifier, LogisticRegression_classifier)


# Parse the input csv file and put lemmatized words into a list
def sentiment(cleaned_script):
    with open(cleaned_script) as csv_f:
        csv_reader = csv.reader(csv_f, delimiter=',')
        for lines in csv_reader:
            script = lines[0]
    feats = find_features(script)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

def main():
    print(sentiment("transcripts_cleaned.csv"))

if __name__ == "__main__":
    main()


