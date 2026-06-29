from src.evaluation.metrics import accuracy


def main():

    print(accuracy(3, 3))
    print(accuracy(2, 3))
    print(accuracy("yes", "yes"))
    print(accuracy("no", "yes"))


if __name__ == "__main__":
    main()