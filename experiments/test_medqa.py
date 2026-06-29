from src.data.medqa import load_medqa


def main():
    dataset = load_medqa()

    print(dataset)

    print("\nFirst Example\n")

    print(dataset[0])


if __name__ == "__main__":
    main()
