import os

def main():
    print("Choose an option:")
    print("1. Train the model")
    print("2. Run the app")

    choice = input("Enter choice (1/2): ")

    if choice == '1':
        os.system('python train_model.py')
    elif choice == '2':
        os.system('streamlit run app.py')
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()