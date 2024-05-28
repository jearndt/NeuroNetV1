import os
import pickle
import pandas as pd

def load_spikes_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        spikes = pickle.load(f)
        # Debug: Check the content of spikes
        print(f"Loaded spikes from {pickle_file}: {spikes}")
    return spikes

def create_dataset_from_pickles(pickle_dir="pickles"):
    dataset = {
        'file_name': [],
        'neuron_id': [],
        'spike_times': []
    }
    
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
    
    for pickle_file in pickle_files:
        pickle_path = os.path.join(pickle_dir, pickle_file)
        spikes = load_spikes_from_pickle(pickle_path)
        
        for neuron_id, neuron_spikes in enumerate(spikes):
            if not neuron_spikes:
                print(f"No spikes for neuron {neuron_id} in file: {pickle_file}")
            dataset['file_name'].append(pickle_file)
            dataset['neuron_id'].append(neuron_id)
            dataset['spike_times'].append(neuron_spikes)
    
    return pd.DataFrame(dataset)

if __name__ == '__main__':
    dataset = create_dataset_from_pickles()
    print(dataset.head())  # Display the first few rows of the dataset
    dataset.to_csv('spike_dataset.csv', index=False)  # Save the dataset to a CSV file
