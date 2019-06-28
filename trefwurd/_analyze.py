import re


PREPOSITIONS = """
aan achter af behalve binnen bij door in langs met na naar naast om onder op over
per sinds tegen tegenover tijdens tot tussen uit via volgens voor zonder
""".split()  # not included: "te"


TREMA_NORMALIZATION_TABLE = str.maketrans(
    {"ë": "e", "ï": "i", "ö": "o", "ü": "u"}
)

# VOWELS
SIMPLE_SHORT_VOWELS = "a e i o u y".split()
SIMPLE_LONG_VOWELS_DOUBLE = "aa ee oo uu".split()
# Can be both long and short, but orthographically behave as long vowels/
FREE_VOWELS = "ie oe eu".split()
PURE_DIPTHONGS = "ei ij ui ou".split()  # 'ij' is always pronounced as 'ei', but it's tricky due  its impurity.
IMPURE_DIPTHONGS = "ai oi aai ooi oei eeuw ieuw".split()

COMPLEX_VOWELS = IMPURE_DIPTHONGS + PURE_DIPTHONGS + FREE_VOWELS

# CONSONANTS
VOICED_OCCLUSIVES = "b d g v z".split()
VOICELESS_OCCLUSIVES = "t k f c s ch p sh sch x".split()  # 't kofschip + x; "sch" is technically cheating
AFFRICATES = "tsj ts ds".split()  # not incl. "pi*zza*" and "*g*in"
NASALS_GLIDES_LATERALS = "m n ng l r w".split()

ALL_CONSONANTS = (VOICELESS_OCCLUSIVES
                  + VOICED_OCCLUSIVES
                  + AFFRICATES
                  + NASALS_GLIDES_LATERALS)

FRICATIVE_DEVOICING_INVERTED_TABLE = str.maketrans(
    {"s": "z", "f": "v"}
)

for l in (PREPOSITIONS,
          IMPURE_DIPTHONGS, VOICELESS_OCCLUSIVES, NASALS_GLIDES_LATERALS, ALL_CONSONANTS,
          COMPLEX_VOWELS):
    l.sort(key=len, reverse=True)


def decomposition_pattern(prefixes=None, suffixes=None):
    return r"""
        (?P<vowel_min4>
            {complex_v}
            |
            ([{simple_v}])\2?
        )?
        (?P<cons_min3>
            ({cons}){{0,1}}?
        )?
        (?P<vowel_min2>
            {complex_v}
            |
            ([{simple_v}])\6?
        )
        (?P<cons_min1>
            ({cons}){{0,2}}?
        )
        (?P<ending>
            ({end})
        )?
        $
        """.format(complex_v="|".join(COMPLEX_VOWELS),
                   simple_v="".join(SIMPLE_SHORT_VOWELS),
                   cons="|".join(ALL_CONSONANTS),
                   end="|".join(suffixes or []))


NONPREPOSITIONAL_PREFIXES = "mee omver bijeen ont on ineen heen weg".split()
VERB_ENDINGS = "ten te den de ende end nd d t en".split()

VERB_PREFIXES = NONPREPOSITIONAL_PREFIXES + PREPOSITIONS

for l in (NONPREPOSITIONAL_PREFIXES, VERB_ENDINGS, VERB_PREFIXES):
    l.sort(key=len, reverse=True)


VERB_PREP_ADP_PREFIX_RE = re.compile(r"({}){{1,3}}".format("|".join(VERB_PREFIXES)))


VERB_ENDING_DECOMPOSITION_RE = re.compile(
    decomposition_pattern(suffixes=VERB_ENDINGS),
    flags=re.VERBOSE)


def decomposition(string, decomposition_regex, prefix_regex, trema_normalize=True):
    """RETURNS (tuple) of (prefixes, nucleus, nucleus_ending_analysis, suffix)"""
    if trema_normalize:
        string = string.translate(TREMA_NORMALIZATION_TABLE)

    prefixes = []
    while string:
        m = prefix_regex.match(string)
        if m:
            prefixes.append(m.group())
            string = string[m.end():]
        else:
            break

    last_components = []
    m = decomposition_regex.search(string)
    if not m:
        suffix = ""
        nucleus = string
    else:
        a = m.groupdict()
        suffix = a.get("ending") or ""
        nucleus = string[:m.start()]
        last_components.extend(x or "" for x in(a.get("vowel_min4"),
                                                a.get("cons_min3"),
                                                a.get("vowel_min2"),
                                                a.get("cons_min1")))

    return prefixes, nucleus, last_components, suffix


def decompose_noun(string):
    pass


def decompose_verb(string):

    constituents = decomposition(string,
                                 VERB_ENDING_DECOMPOSITION_RE,
                                 VERB_PREP_ADP_PREFIX_RE)

    prefixes, nucleus, last_comps, suffix = constituents

    # check if string starts with prep or adp
    if suffix in ("d", "t"):
        if nucleus.startswith("ge"):
            nucleus_left = nucleus[2:]
        # if no final consonant cluster before word suffix
        if not last_comps[-1]:
            last_comps.append(suffix)

        # if (verb_form_suff
        # and final_vowel not in IMPURE_DIPTHONGS
        # and not final_cons_cluster):
        # stems.append((prep_adp_prefix, stem + verb_form_suff[0]))

    return (prefixes, nucleus, last_comps)


def partition_hypotheses(string, word_class):
    """Split word form into left and right side"""
    pass



print(decomposition("onderuitgezet", VERB_ENDING_DECOMPOSITION_RE, VERB_PREP_ADP_PREFIX_RE))
print(decompose_verb("onderuitgezet"))

