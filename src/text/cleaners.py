""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Regular expression matching whitespace:
import re
from unidecode import unidecode
from src.text.number_cleaner import normalize_numbers

_whitespace_re = re.compile(r'\s+')
# \b(%s)(s?)\b[\.]?
_abbreviations = [(re.compile('\\b(%s)(s?)\\b[\\.]?' % x[0], re.IGNORECASE), x[1]) for x in [
    ('lbbb', 'left bundle branch block'),
    ('rbbb', 'right bundle branch block'),
    ('pvc', "premature ventricular complex"),
    ('pac', 'premature atrial complex'),
    ('lvh', 'left ventricular hypertropy'),
    ('rvh', 'right ventricular hypertropy'),
    ('ivcd', "intralventricular conduction delay "),
    ('lad', 'left axis deviation'),
    ("rad", "right axis deviation"),
    ("avnrt", "av-nodal re-entry tachycardia"),
    ("svt", "supraventricular tachycardia"),
    ("chb", "complete heart block"),
    ("mbbb", "masquerading bundle branch block"),
    ("aivr", "accelerated idioventricular rhythm"),
    ("ajr", "accelerated junctional rhythm"),
    ("arvc", "arrhythmogenic right ventricular cardiomyopathy"),
    ("af", "atrial fibrillation"),
    ("afib", "atrial fibrillation"),
    # ("bpm", "beat per minute"),
    # ("ms", "millisecond"),
    ("lpfb", "left posterior fascicular block"),
    ("lvh", "left ventricular hypertrophy"),
    ("lmca", "left main coronary artery"),
    ("cpvt", "catecholamine polymorphic ventricular tachycardia"),
    ("pvt", "polymorphic ventricular tachycardia"),
    ("lqts", "long qt syndrome"),
    ("lpfb", "left posterior fascicular block"),
    ("prwp", "Poor R-wave progression"),
    ('rvot', "right ventricular outflow tract"),
    ("tdp", "Torsades de pointes"),
    ("ami", "anterior myocardial infarction"),
    ("lafb", "left anterior fascicular block"),
    ("lpdb", "Left Posterior Fascicular Block"),
    ("wpw", "wolf parkinson white")
]]

replacements = [
    ("INTERPRETATION MADE WITHOUT KNOWLEDGE OF PATIENT'S SEX AND AGE", ""),
    ("(s)", ""),
    (" - age undetermined", ""),
    (" - no further analysis", ""),
    ("°", " degree"),
]

def expand_abbreviations(text):
    for acronym, full_noun in _abbreviations:
        text = acronym.sub(full_noun, text)
    return text

def expand_replace(text):
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def lowercase(text):
    return text.lower()

def expand_numbers(text):
    return normalize_numbers(text)

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    return unidecode(text)

def ecg_text_cleaner(text):
    text = expand_replace(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    return text.strip()

def english_cleaner(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text.strip()


if __name__ == "__main__":
    text = 'affect af rbbb pvc(s) pacs 10 bpm'
    text = ecg_text_cleaner(text)
    print(text)