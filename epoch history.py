import pickle

# Load the existing training history
history_path = 'training_history.pkl'
with open(history_path, 'rb') as file:
    history_data = pickle.load(file)

print(history_data)
# Add the 'epoch' key if it's missing, and set it to range 1 to 20
if 'epoch' not in history_data:
    history_data['epoch'] = list(range(1, 21))  # Assuming your past training covered 20 epochs
    print(f"Added epoch range from 1 to 20.")

# Save the updated history back to the file
with open(history_path, 'wb') as file:
    pickle.dump(history_data, file)

print("Updated training history saved successfully.")
