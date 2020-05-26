import nltk, random, pickle
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# Load two sample texts
short_pos = open("sample_texts/positive.txt", "r", encoding = "ISO-8859-1").read()
short_neg = open("sample_texts/negative.txt", "r", encoding = "ISO-8859-1").read()

# Tokenize each text and label each word
'''
TO DO:
    find sample texts for other sentiment
'''

all_words = []
sample_text = []

allowed_word_types = ["J", "N"] # 'J' -> Adjective, 'N' -> Noun, 'R' -> Adverb, 'V' -> Verb

# Label each word as pos in positive.txt
for p in short_pos.split('\n'):
    sample_text.append((p, "pos"))
    words = word_tokenize(p)
    # Tag each word and generate a list of tuples in the form of ('Word', 'Type')
    pos = nltk.pos_tag(words)
    for w in pos:
        # The first letter of a tag indicates its general type
        # For example:
        # "NN" -> noun, common, sigular or mass, "NNS" -> noun, common, plural
        if w[1][0] in allowed_word_types:
            # Collect words with desired types in lowercase letter
            all_words.append(w[0].lower())

# Label each word as pos in negative.txt
for p in short_neg.split('\n'):
    sample_text.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

# Save the tagged words with pickle
save_text = open("pickled_algos/sample_text.pickle","wb")
pickle.dump(sample_text, save_text)
save_text.close()

# Pick the 5000 most frequent words and save it with pickle
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]
save_word_features = open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(text):
    '''
    Tokenize input text and generates a featureset showing if words
    in word_features can be found in the input text
    '''
    words = word_tokenize(text)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in sample_text]
random.shuffle(featuresets)
setsize = len(featuresets)

# Divide the list of featuresets into traning set and testing set 
testing_set = featuresets[int(0.8 * setsize):]
training_set = featuresets[:int(0.8 * setsize)]

# Train eachc lassifier and save the trained algorithms using pickle
NaiveBayes_classifier = nltk.NaiveBayesClassifier.train(training_set)
print("NaiveBayes_classifier accuracy:", (nltk.classify.accuracy(NaiveBayes_classifier, testing_set))*100)
save_classifier = open("pickled_algos/NaiveBayes_classifier.pickle","wb")
pickle.dump(NaiveBayes_classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
save_classifier = open("pickled_algos/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
save_classifier = open("pickled_algos/BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
save_classifier = open("pickled_algos/LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)
save_classifier = open("pickled_algos/SGDC_classifier.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

# Print the most informative features
# classifier.show_most_informative_features(15)
