from automathon import DFA
import string
import random

def quote_special(char):
    if char.isalnum():
        return char
    return f'"{char}"'

def create_english_dfa():
    # q0: start, q1: started with lowercase (and continues lowercase),
    # q2: started with uppercase (first char only) and continues with lowercase,
    # q_reject: sink for invalid inputs
    q = {'q0', 'q1', 'q2', 'q_reject'}

    lowercase_letters = set(string.ascii_lowercase)
    uppercase_letters = set(string.ascii_uppercase)
    numbers = set(string.digits)
    special_chars_to_quote = {' ', '.', ',', '!', '?', '_', '-', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', ':', ';', '"', "'", '<', '>', '/', '~', '`'}

    special_chars = set(quote_special(c) for c in special_chars_to_quote)
    
    sigma = lowercase_letters | uppercase_letters | numbers | special_chars

    delta = {}

    #define transitions from all the states
    delta['q0'] = {}
    # from start: a lowercase leads to q1, an uppercase leads to q2 (allowed only as first char)
    for char in lowercase_letters:
        delta['q0'][char] = 'q1'
    for char in uppercase_letters:
        delta['q0'][char] = 'q2'
    # any digit or special at start -> reject
    for char in numbers | special_chars:
        delta['q0'][char] = 'q_reject'

    # q1: seen lowercase at first position (or continuing lowercase)
    delta['q1'] = {}
    for char in lowercase_letters:
        delta['q1'][char] = 'q1'
    # uppercase anywhere after first char -> reject
    for char in uppercase_letters | numbers | special_chars:
        delta['q1'][char] = 'q_reject'

    # q2: first char was uppercase, remaining must be lowercase only
    delta['q2'] = {}
    for char in lowercase_letters:
        delta['q2'][char] = 'q2'
    # any uppercase after first, numbers or special -> reject
    for char in uppercase_letters | numbers | special_chars:
        delta['q2'][char] = 'q_reject'
    
    delta['q_reject'] = {}
    for char in sigma:
        delta['q_reject'][char] = 'q_reject'

    initial_state = 'q0'

    # accept words that started with lowercase (q1) or started with uppercase then continued lowercase (q2)
    f = {'q1', 'q2'}

    return DFA(q, sigma, delta, initial_state, f)

def test_input(automata, word):
    try:
        if automata.accept(word):
            return "Accepted"
        else:
            return "Not Accepted"
    except Exception:
        return "Not Accepted"


def generate_test_words(per_category=5, max_len=8, seed=42):
    """Generate a variety of test words to exercise the DFA.

    Categories generated:
    - valid_lower: all-lowercase words
    - valid_initial_cap: initial uppercase then lowercase
    - invalid_upper_later: uppercase appears after first char
    - invalid_contains_digit: digit anywhere
    - invalid_starts_digit: starts with digit
    - invalid_special: contains a special character
    - leading_space: starts with a space
    - empty and all_uppercase
    """
    rnd = random.Random(seed)
    lowers = string.ascii_lowercase
    uppers = string.ascii_uppercase
    digits = string.digits
    specials = " _-@.!?,'"  # include a small set of specials used in tests

    out = []

    def rand_word(chars, min_len=1, max_len=max_len):
        length = rnd.randint(min_len, max_len)
        return ''.join(rnd.choice(chars) for _ in range(length))

    # valid lowercase words
    for _ in range(per_category):
        out.append(rand_word(lowers))

    # valid initial capital (one uppercase then lowercase)
    for _ in range(per_category):
        first = rnd.choice(uppers)
        rest = rand_word(lowers, min_len=0)
        out.append(first + rest)

    # invalid: uppercase later
    for _ in range(per_category):
        w = rand_word(lowers)
        pos = rnd.randint(1, max(1, len(w)-1)) if len(w) > 1 else 1
        w = w[:pos] + rnd.choice(uppers) + w[pos+1:]
        out.append(w)

    # invalid: contains digit
    for _ in range(per_category):
        w = rand_word(lowers)
        pos = rnd.randint(0, len(w))
        w = w[:pos] + rnd.choice(digits) + w[pos:]
        out.append(w)

    # invalid: starts with digit
    for _ in range(per_category):
        w = rnd.choice(digits) + rand_word(lowers)
        out.append(w)

    # invalid: contains special
    for _ in range(per_category):
        w = rand_word(lowers)
        pos = rnd.randint(0, len(w))
        w = w[:pos] + rnd.choice(specials) + w[pos:]
        out.append(w)

    # leading space
    for _ in range(max(1, per_category//2)):
        out.append(' ' + rand_word(lowers))

    # empty and all uppercase
    out.append("")
    for _ in range(max(1, per_category//2)):
        out.append(rand_word(uppers))

    return out
    
def main():

    automata = create_english_dfa()

    test_cases = [
        "cat",
        "dog", 
        "a",
        "zebra",
        "hello",
        "world",
        "programming",

        "dog1",        # contains digit
        "1dog",        # starts with digit
        "DogHouse",    # contains uppercase letters
        "Dog_house",   # contains underscore
        " cats",       # starts with space
        "Cat",         # starts with uppercase
        "hello world", # contains space
        "test@email",  # contains special character
        "",            # empty string
        "123",         # all digits
        "HELLO",       # all uppercase
        "hello!",
    ]

    # generate additional test words and append
    generated = generate_test_words(per_category=6, max_len=8, seed=123)
    # mark generated cases so it's easy to identify in output
    generated_tagged = [f"[GEN]{w}" for w in generated]
    test_cases.extend(generated)

    output_lines = []
    for word in test_cases:
        result = test_input(automata, word)
        output_lines.append(f"Input: {word}, Result: {result}")

    # append a small summary of generated tests
    output_lines.append("")
    output_lines.append("Generated test words:")
    for w in generated:
        output_lines.append(w)

    # Write results to Lab2/dfa1_output.txt
    output_path = "Lab2/dfa1_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    # Generate visualization in Lab2 directory
    print("Results are stored in ")
    print("Generating Visualization")
    automata.view("Lab2/English_Word_DFA")

if __name__ == "__main__":
    main()