import spacy

# load the saved model
nlp = spacy.load("textcat_model")

# use the model to predict a new company
doc = nlp("NewCompanyName")
print(doc.cats)
