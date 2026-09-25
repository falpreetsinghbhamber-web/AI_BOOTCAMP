!pip install spacy

!python -m spacy download en_core_web_sm

import spacy

# Load spaCy's lightweight English core model (~12MB)
nlp = spacy.load("en_core_web_sm")

# Sample unstructured text input
raw_resume_text = """
Sneha is a Machine Learning Engineer living in Delhi.
She worked at Google for 5 years starting from August 2018.
Sneha graduated from SRM University and has expertise in TensorFlow, Deep Learning and NLP.
"""

print("Processing text with spaCy...\n")
doc = nlp(raw_resume_text)

print("--- Extracted Named Entities ---")
# Extract entities like Persons, Organizations, Locations, and Dates
for ent in doc.ents:
  print(f"Entity: {ent.text:<20} | Type: {ent.label_:<10} ({spacy.explain(ent.label_)})")

print("\n--- Identified Keywords/Nouns ---")
# Extract key nouns/skills
nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
print("Key Nouns Found:", list(set(nouns)))
