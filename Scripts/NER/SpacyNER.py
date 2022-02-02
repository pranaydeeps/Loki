# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy
import os
import pandas as pd

def main():
        # Load English tokenizer, tagger, parser and NER
        nlp = spacy.load("en_core_web_trf")

        MemeTextPath = os.path.join("..", "textAllEntities.csv")
        MemeTextDF = pd.read_csv(MemeTextPath)
        MemeTextOnly = MemeTextDF["OCR"].tolist()
        print(MemeTextOnly[:5])

        for memetext in MemeTextOnly[:5]:
                doc = nlp(memetext)
                # Find named entities, phrases and concepts
                for entity in doc.ents:
                        if str(entity.label_) != "DATE":
                                print(entity.text, entity.label_)


if __name__ == "__main__":
        main()
        print("Complete")