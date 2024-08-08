import numpy as np
from sklearn.tree import DecisionTreeClassifier

class BigramModel:
    def _init_(self):
        self.classifier = None
        self.bigram_to_index = {}

    def train(self, words):
        """
        Train the model using a list of words.
        :param words: List of words to train the model.
        """
        
        bigrams = {word[i:i + 2] for word in words for i in range(len(word) - 1)}
        sorted_bigrams = sorted(bigrams)
        self.bigram_to_index = {bigram: index for index, bigram in enumerate(sorted_bigrams)}

       
        num_words = len(words)
        num_bigrams = len(sorted_bigrams)
        feature_matrix = np.zeros((num_words, num_bigrams), dtype=int)
        labels = np.array(words)

        i = 0
        while i < len(words):
            word = words[i]
            j = 0
            while j < len(word) - 1:
                bigram = word[j:j + 2]
                if bigram in self.bigram_to_index:
                    feature_matrix[i, self.bigram_to_index[bigram]] = 1
                j += 1
            i += 1

        
        self.classifier = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_split=5)
        self.classifier.fit(feature_matrix, labels)

    def predict(self, input_bigrams):
        """
        Predict possible words given a list of bigrams.
        :param input_bigrams: List of bigrams.
        :return: List of predicted words.
        """
        if not self.classifier:
            raise ValueError("Error")

        
        prediction_vector = np.zeros((1, len(self.bigram_to_index)), dtype=int)
        i = 0
        while i < len(input_bigrams):
            bigram = input_bigrams[i]
            if bigram in self.bigram_to_index:
                prediction_vector[0, self.bigram_to_index[bigram]] = 1
            i += 1

        
        probabilities = self.classifier.predict_proba(prediction_vector)[0]
        sorted_indices = np.argsort(probabilities)[::-1]
        candidates = [self.classifier.classes_[idx] for idx in sorted_indices]

      
        predictions = [word for word in candidates if all(b in word for b in input_bigrams)]
        return predictions[:5]


bigram_model = BigramModel()

################################
# Non Editable Region Starting #
################################
def my_fit(words):
    bigram_model.train(words)
    return bigram_model
################################
#  Non Editable Region Ending  #
################################

################################
# Non Editable Region Starting #
################################
def my_predict(model, bigram_list):
    return model.predict(bigram_list)
################################
#  Non Editable Region Ending  #
################################