from src.data.loader import load_dataset


def main():

    for name in [
        "medqa",
        "medmcqa",
        "pubmedqa",
    ]:

        dataset = load_dataset(name)

        print("=" * 80)
        print(name.upper())
        print(dataset[0])


if __name__ == "__main__":
    main()