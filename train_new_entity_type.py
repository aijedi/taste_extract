#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type
This script shows how to add a new entity type to an existing pre-trained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.
The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.
After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# new entity label
LABEL = 'TASTE'

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [

( u"To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora." ,{'entities':[(73,89, u'TASTE' )]}),

( u"The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not." ,{'entities':[(12,33, u'TASTE' )]}),

( u"Not only was the food outstanding, but the little 'perks' were great." ,{'entities':[(22,33, u'TASTE' )]}),

( u"Not only was the food outstanding, but the little 'perks' were great." ,{'entities':[(63,68, u'TASTE' )]}),

( u"It is very overpriced and not very tasty." ,{'entities':[(26,40, u'TASTE' )]}),

( u"Our agreed favorite is the orrechiete with sausage and chicken (usually the waiters are kind enough to split the dish in half so you get to sample both meats)." ,{'entities':[(11,19, u'TASTE' )]}),

( u"The Bagels have an outstanding taste with a terrific texture, both chewy yet not gummy." ,{'entities':[(19,30, u'TASTE' )]}),

( u"Nevertheless the food itself is pretty good." ,{'entities':[(32,43, u'TASTE' )]}),

( u"They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it." ,{'entities':[(30,36, u'TASTE' )]}),

( u"The pizza is the best if you like thin crusted pizza." ,{'entities':[(17,21, u'TASTE' )]}),

( u"We were very disappointed." ,{'entities':[(13,25, u'TASTE' )]}),

( u"IT IS DEFINITELY SPECIAL AND AFFORDABLE." ,{'entities':[(17,24, u'TASTE' )]}),

( u"From the incredible food, to the warm atmosphere, to the friendly service, this downtown neighborhood spot doesn't miss a beat." ,{'entities':[(9,19, u'TASTE' )]}),

( u"Great food at REASONABLE prices, makes for an evening that can't be beat!" ,{'entities':[(0,5, u'TASTE' )]}),

( u"The fried rice is amazing here." ,{'entities':[(18,25, u'TASTE' )]}),

( u"Three courses - choices include excellent mussels, puff pastry goat cheese and salad with a delicious dressing, and a hanger steak au poivre that is out of this world." ,{'entities':[(32,41, u'TASTE' ),(92,101, u'TASTE' )]}),

( u"it's a perfect place to have a amazing indian food." ,{'entities':[(12,33, u'TASTE' )]}),

( u"At the end you're left with a mild broth with noodles that you can slurp out of a cup." ,{'entities':[(30,34, u'TASTE' )]}),
( u"I just wonder how you can have such a delicious meal for such little money." ,{'entities':[(38,47, u'TASTE' )]}),

( u"The food was delicious but do not come here on a empty stomach." ,{'entities':[(13,22, u'TASTE' )]}),

( u"Ive been to many Thai restaurants in Manhattan before, and Toons is by far the best Thai food Ive had (except for my mom's of course)." ,{'entities':[(79,83, u'TASTE' )]}),

( u"Nice atmosphere, the service was very pleasant and the desert was good." ,{'entities':[(66,70, u'TASTE' )]}),

( u"Fabulous service, fantastic food, and a chilled out atmosphere and environment." ,{'entities':[(18,27, u'TASTE' )]}),

( u"Great food, good size menu, great service and an unpretensious setting." ,{'entities':[(0,5, u'TASTE' )]}),

( u"The menu is limited but almost all of the dishes are excellent." ,{'entities':[(53,62, u'TASTE' )]}),

( u"Unfortunately, the food is outstanding, but everything else about this restaurant is the pits." ,{'entities':[(27,38, u'TASTE' )]}),

( u"We always have a delicious meal and always leave feeling satisfied." ,{'entities':[(17,26, u'TASTE' )]}),

( u"The pizza was pretty good and huge." ,{'entities':[(14,25, u'TASTE' )]}),

( u"The atmosphere is unheralded, the service impecible, and the food magnificent." ,{'entities':[(66,77, u'TASTE' )]}),

( u"The wait staff is friendly, and the food has gotten better and better!" ,{'entities':[(52,58, u'TASTE' )]}),

( u"It may be a bit packed on weekends, but the vibe is good and it is the best French food you will find in the area." ,{'entities':[(71,75, u'TASTE' )]}),

( u"Right off the L in Brooklyn this is a nice cozy place with good pizza." ,{'entities':[(59,63, u'TASTE' )]}),

( u"We had the lobster sandwich and it was FANTASTIC." ,{'entities':[(39,48, u'TASTE' )]}),

( u"Deep Fried Skewers are good and still rare to find in NYC." ,{'entities':[(23,27, u'TASTE' )]}),

( u"Their tuna tartar appetizer is to die for." ,{'entities':[(34,41, u'TASTE' )]}),

( u"An oasis of refinement:  Food, though somewhat uneven, often reaches the pinnacles of new American fine cuisine - chef's passion (and kitchen's precise execution) is most evident in the fish dishes and soups." ,{'entities':[(47,53, u'TASTE' )]}),

]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, new_model_name='animal', output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    nlp = spacy.load('en')  # create blank Language class
    print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()



    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = 'The food always tastes fresh and served promptly.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is None:
        output_dir = Path('/home/tanush/Desktop/NER Project/Taste Extractor/en_rev_taste')
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)
