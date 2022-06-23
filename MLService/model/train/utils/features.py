from utils.aux_func import generate_bigrams, search_sequence_of_words, search_words, get_special_bigrams
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load() #spacy.load("en_core_web_sm")

# Get third person function.
def get_third_person(text: str, bigram_window: int = 5, verbose = False):
    """
    This functions determines if the text is written in third person or not.
    Input: text (str)
    Output: Boolean (true => it is written in third person, false => is not written in third person)
    """

    # Third person pronouns.
    THIRD_PERSON = ["she", "her", "hers", "herself"]

    # Generate third-person bigrams.
    tp_bigram_one = ["new", "young", "beautiful", "different", "open-minded", 
                    "hottest", "sizzling", "sexy", "gorgeous", "few", "busty", "exotic"]
                    
    tp_bigram_two = ["girl", "girls", "ladie", "ladies", "dancer", "dancers", 
                    "chick", "chicks", "playmates", "babes"]

    TP_SPECIAL_BIGRAMS = generate_bigrams(tp_bigram_one, tp_bigram_two)
    TP_OTHER_BIGRAMS = ["all nationalities"]
    TP_SPECIAL_BIGRAMS = TP_SPECIAL_BIGRAMS + TP_OTHER_BIGRAMS

    # The text is written in third person singular or in third person plural.
    # or contains special word.
    try:
      doc = nlp(text)
    except:
      return int(False)
    b1 = search_words(doc, THIRD_PERSON, verbose = verbose)
    b2 = get_special_bigrams(doc, TP_SPECIAL_BIGRAMS, bigram_window = bigram_window, verbose = verbose)
    #print(b1, b2, b3)
    return int(b1 or b2)

# Get first person plural function.
def get_first_person_plural(text: str, bigram_window: int = 5, verbose = False):
    """
    This functions determines if the text is written in third person or not.
    Input: text (str)
    Output: Boolean (true => it is written in third person, false => is not written in third person)
    """
    # First person plural pronouns
    FIRST_PERSON_PLURAL = ["us", "our", "ours", "ourselves"]

    # Generate third-person bigrams.
    fp_bigram_one = ["with", "message", "visit"]
    fp_bigram_two = ["us"] 
    FP_SPECIAL_BIGRAMS = generate_bigrams(fp_bigram_one, fp_bigram_two)
    FP_OTHER_BIGRAMS = ["our website"]
    FP_SPECIAL_BIGRAMS = FP_SPECIAL_BIGRAMS + FP_OTHER_BIGRAMS

    # The text is written in third person singular or in third person plural.
    # or contains special word.
    try:
      doc = nlp(text)
    except: 
      return int(False)
    b1 = search_words(doc, FIRST_PERSON_PLURAL, verbose=verbose)
    b2 = get_special_bigrams(doc, FP_SPECIAL_BIGRAMS, bigram_window = bigram_window, verbose=verbose)
    #print(b1, b2)
    return int(b1 or b2)

# Get human trafficking keywords.
def ht_find_keywords(text: str, verbose: bool = False):
    HT_KEYWORDS = ["new in town", "new girl", "came to the town", "out off town", 
               "few days",  "back to town", "out of town", "in your town", 
               "for a few days", "back in town",  "only stay for a few", "new arrived", 
               "just arrived", "new here", "new to this",  "new agency", "new in town", 
               "new to your city", "i'm new", "i am new", "brand new",  "little new",
               "very new", "look new", "short-term", "new to the area", "new to the industry", 
               "new in the business", "new face", "short time", "in town for", "new to town", 
               "short stay", "short term", "short time", "short period", "arrived in town", 
               "arrived short term", "for a limited time", "newly arrived", "passing through", 
               "on vacation", "in town for a few days", "i am a newb", "i am a newbie", "first time to", 
               "first time in", "first time here", "first time visiting", "new in the field",
               "just turned 18", "turned 18", "hot teen"]
    try:
      doc = nlp(text)
    except: 
      return int(False)
    b1 = search_sequence_of_words(text, HT_KEYWORDS, verbose = verbose) 
    # print(b1, b2)
    return int(b1)

# Service is restricted somehow.
def service_is_restricted(text: str, verbose: bool = False):
    WITH_CONDOM_SEQUENCE = ["with condom", "use of condoms", "with a condom", "no bb"]
    RESTRICTED_SEX_SEQUENCE = ["no oral", "no anal", "no black", "no greek"]

    try:
      doc = nlp(text)
    except: 
      return int(False)
    b1 = search_sequence_of_words(text, WITH_CONDOM_SEQUENCE, verbose = verbose) 
    b2 = search_sequence_of_words(text, RESTRICTED_SEX_SEQUENCE, verbose = verbose) 
    return int(b1 or b2)

# Determine if service is offer incall
def offer_incall(text: str, verbose: bool = False):
    INCALL_WORDS = ["incall", "in-call", "incalls"]
    SPECIAL_INCALL_SEQUENCE = ["in call", "in call only", "in-call only", "incall only"]
    SPECIAL_NOT_INCALL_SEQUENCE = ["no incalls", "no incall", "no in-calls", "no in calls", "no incall"]

    try:
      doc = nlp(text)
    except: 
      return int(False)
    b1 = search_words(doc, INCALL_WORDS, verbose = verbose)
    b2 = search_sequence_of_words(text, SPECIAL_INCALL_SEQUENCE, verbose = verbose) 
    b3 =  search_sequence_of_words(text, SPECIAL_NOT_INCALL_SEQUENCE, verbose = verbose)
    # print(b1, b2)
    return (b1 or b2) and not b3
      
# Determine if service is offer outcall
def offer_outcall(text: str, verbose: bool = False):
    OUTCALL_WORDS = ["outcall", "out-call", "outcalls"]
    SPECIAL_OUTCALL_SEQUENCE = ["out call", "out call only", "out-call only", "outcall only"]
    SPECIAL_NOT_OUTCALL_SEQUENCE = ["no outcall", "no out call", "no out-call"]

    try:
      doc = nlp(text)
    except: 
      return False
    b1 = search_words(doc, OUTCALL_WORDS, verbose = verbose)
    b2 = search_sequence_of_words(text, SPECIAL_OUTCALL_SEQUENCE, verbose = verbose)
    b3 =  search_sequence_of_words(text, SPECIAL_NOT_OUTCALL_SEQUENCE, verbose = verbose)
    #print(b1, b2)
    return (b1 or b2) and not b3

# Get service place.
def service_place(text: str, verbose: bool = False):
    incall = offer_incall(text, verbose = verbose)
    outcall = offer_outcall(text, verbose = verbose)
    if incall and not outcall:
      return 1
    if (incall and outcall) or outcall:
      return 0
    # Missing values.
    return -1