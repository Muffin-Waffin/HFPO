from src.evaluation.parser import (
    parse_multiple_choice,
    parse_yes_no_maybe,
)


def main():

    print(parse_multiple_choice("The answer is B."))
    print(parse_multiple_choice("Option D is correct."))
    print(parse_multiple_choice("C. Ceftriaxone"))

    print(parse_yes_no_maybe("Yes"))
    print(parse_yes_no_maybe("The answer is no."))
    print(parse_yes_no_maybe("Maybe"))


if __name__ == "__main__":
    main()