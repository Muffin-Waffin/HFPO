from src.data.medmcqa import load_medmcqa

def main():
    dataset = load_medmcqa()

    print(dataset)

    print("\nFirst Example\n")

    print(dataset[0])


if __name__ == "__main__":
    main()