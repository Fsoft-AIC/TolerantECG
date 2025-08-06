""" from https://github.com/keithito/tacotron """

import re

import inflect

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")
_sign_re = re.compile(r"[+-]?\d+")
_range_re = re.compile(r"(\d+)-(\d+)\b")
_v_range_re = re.compile(r"v(\d+)-(\d+)", re.IGNORECASE)

def _remove_commas(m):
    return m.group(1).replace(",", "")

def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")

def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))

def _expand_sign(m):
    match = m.group()
    if match.startswith("-"):
        return "%s %s" % ("minus", match[1:])
    elif match.startswith("+"):
        return match[1:]
    else:
        return match

def _expand_v_range(m):
    start, end = m.groups()
    return "V %s to %s" % (start, end)

def _expand_range(m):
    start, end = m.groups()
    return "%s to %s" % (start, end)

def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100).replace("-", " ")
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100).replace("-", " ") + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ").replace("-", " ")
    else:
        return _inflect.number_to_words(num, andword="").replace("-", " ")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_v_range_re, _expand_v_range, text)
    text = re.sub(_range_re, _expand_range, text)
    text = re.sub(_sign_re, _expand_sign, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

if __name__ == "__main__":
    print(normalize_numbers("it is -30 to +60 degree, v1-6, 50-120 bpm"))