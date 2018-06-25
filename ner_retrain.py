#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.
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
#import en_core_taste

# training data
TRAIN_DATA = [

(u"The bruscetta is a bit soggy, but the salads were fresh, included a nice mix of greens (not iceberg) all dishes are served piping hot from the kitchen.",{'entities':[(19,28,u'TASTE'),(50,55,u'TASTE'),(123,133,u'TASTE')]}),

(u"And evaluated on those terms Pastis is simply wonderful.",{'entities':[(46,55,u'TASTE')]}),

(u"The atmosphere isn't the greatest , but I suppose that's how they keep the prices down .",{'entities':[ ]}),

(u"The pickles were great addition.",{'entities':[(17,22,u'TASTE')]}),

(u"If your favorite Chinese food is General Tao chicken, then this is NOT your place.",{'entities':[ ]}),

(u"You must try the shrimp appetizers.",{'entities':[ ]}),

(u"Not only is the cuisine the best around, the service has always been attentive and charming.",{'entities':[(28,32,u'TASTE')]}),

(u"Filled with suits, though, so you never really feel terribly comfortable.",{'entities':[ ]}),

(u"The production is a symphony, alot of fun to experience.The food sublime for the most part.",{'entities':[(65,72,u'TASTE')]})

]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model='/home/tanush/Desktop/NER Project/Taste Extractor/en_rev_taste', output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load('en')  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.load('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        # print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is None:
        output_dir = Path('/home/tanush/Desktop/NER Project/Taste Extractor/en_rev_taste')
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            # print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    plac.call(main)
