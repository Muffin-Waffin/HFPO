from src.data.pubmedqa import load_pubmedqa


def main():
    dataset = load_pubmedqa()

    print(dataset)

    print("\nFirst Example\n")

    print(dataset[0])


if __name__ == "__main__":
    main()
