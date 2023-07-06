import spacy
import pandas as pd
from spacy.util import minibatch
from spacy.training import Example
import random

# load data
data = pd.read_csv('training_data.csv')

# convert data into format for training
train_data = []
for i in range(len(data)):
    train_data.append((data.loc[i, 'Account Name'], {"cats": {data.loc[i, 'Other Type']: 1}}))

# create an empty model
nlp = spacy.blank("en")

# add the text classifier to the pipeline
textcat = nlp.add_pipe("textcat")

# add labels to the text classifier
textcat.add_label("Law Firm")
textcat.add_label("School")
textcat.add_label("Vendor")
textcat.add_label("Investment bank/broker")

# train the model
random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.initialize()

losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    batches = minibatch(train_data, size=8)
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(losses)

# save the model
nlp.to_disk("textcat_model")
