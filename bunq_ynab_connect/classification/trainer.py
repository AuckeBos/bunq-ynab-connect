class Trainer:
    def train(self):
        classifier = self.select_best_classifier()
        parameters = self.select_best_parameters(classifier)
        self.train_classifier(classifier, parameters)
