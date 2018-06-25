import spacy
import plac
import random
from pathlib import Path

TRAIN_DATA =[

u"To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora.",

u"Not only was the food outstanding, but the little 'perks' were great.",

u"It is very overpriced and not very tasty."
] 


output_dir = Path('/home/tanush/Desktop/NER Project/Taste Extractor/en_rev_taste')
# test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
for text in TRAIN_DATA:
	doc = nlp2(text)
	print('Entities', [(ent.text, ent.label_) for ent in doc.ents])


          
