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


# Parse the input and put lemmatized words into a list
def get_sentiment(text, text_type, word_features, voted_classifier):
    '''
    
    '''
    
    if text_type == "csv":
        with open(text) as csv_f:
            csv_reader = csv.reader(csv_f, delimiter=',')
            for lines in csv_reader:
                script = lines[0]

    if text_type == "txt":
        text = word_tokenize(documents)

    features = {}
    for w in word_features:
        features[w] = (w in text)

    return voted_classifier.classify(features),voted_classifier.confidence(features)
    

def main():
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

    # Load the saved featureset
    word_features_f = open("pickled_algos/word_features.pickle", "rb")
    word_features = pickle.load(word_features_f)
    word_features_f.close()

    # Create a new VoteClassifier with 5 classifiers
    voted_classifier = VoteClassifier(NaiveBayes_classifier, LinearSVC_classifier,
                                      MNB_classifier, BernoulliNB_classifier,
                                      LogisticRegression_classifier)
    
    print(get_sentiment("transcripts_cleaned.csv", "csv", word_features, voted_classifier))

if __name__ == "__main__":
    main()


