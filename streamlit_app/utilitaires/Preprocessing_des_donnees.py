import numpy as np
import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split


def preprocesser_les_donnees_1(preprocessing_dir, input_data):

    # Create the directory if it doesn't exist
    os.makedirs(preprocessing_dir, exist_ok=True)
    print(f"Directory {preprocessing_dir} created for preprocessing.")


    # Vérifier si la fréquence est constante dans les fichiers raw_.
    # Convert 'Time' column to datetime
    input_data['Time'] = pd.to_datetime(input_data['Time'])

    # Calculate the time difference between consecutive rows
    input_data['Time_diff'] = input_data['Time'].diff()

    # Check if all time differences are equal to 1 second
    consistent_frequency = input_data['Time_diff'].iloc[1:].eq(pd.Timedelta(seconds=1)).all()

    if consistent_frequency:
        st.write("Le jeu de donnée a la même fréquence d'une seconde entre toutes les lignes.")
    else:
        st.write("Le jeu de donnée N'A PAS la même fréquence d'une seconde entre toutes les lignes.")

    # Calculate the time difference between consecutive rows
    input_data['Time_diff'] = input_data['Time'].diff()

    # Sum the time where the time difference is not 1 second
    non_constant_frequency = input_data['Time_diff'].iloc[1:].ne(pd.Timedelta(seconds=1)).sum()
    # Count the number of cases where the time difference is not 1 second
    num_rows_long_diff = len(input_data[input_data['Time_diff'] > pd.Timedelta(seconds=1)])

    st.write(f"Sum of time where the frequency is not constant: {non_constant_frequency}")
    st.write(f"Number of rows with time difference not equal to 1 second: {num_rows_long_diff}")

    # Filter rows where the time difference is greater than 1 second
    long_diff = input_data[input_data['Time_diff'] > pd.Timedelta(seconds=1)]

    # Create a table with the two times and their differences
    table = long_diff[['Time', 'Time_diff']]

    # Display the table
    print(table)

    # Count duplicate timestamps
    duplicate_timestamps = input_data[input_data.duplicated(subset='Time', keep=False)]

    # Display the count of duplicate timestamps
    num_duplicates = duplicate_timestamps.shape[0]
    st.write(f"Number of lines with duplicate timestamps: {num_duplicates}")

    st.write(duplicate_timestamps)

    # Sum of the 'Bits/s' column
    sum_bits_per_second = input_data['Bits/s'].sum()

    st.write(f"Sum of the 'Bits/s' column: {sum_bits_per_second}")

    # Group by 'Time' and sum 'Bits/s' and 'Time_diff' for each timestamp
    # Recompute Time_diff would be even more safe.

    raw_train_g = input_data.groupby('Time').agg({
        'Bits/s': 'sum',  # Sum the 'Bits/s' values for each group
        'Time_diff': 'sum'  # Sum the 'Time_diff' values for each group
    }).reset_index()

    # Display the result
    print(raw_train_g)

    # Sum of the 'Bits/s' column
    sum_bits_per_second_g = raw_train_g['Bits/s'].sum()
    st.write(f"Sum of the 'Bits/s' column: {sum_bits_per_second_g}")

    #Checkconsistency of sum_bits_per_second after transformation
    if (sum_bits_per_second==sum_bits_per_second_g):
        st.write("SUCCES du préprocessing des duplicates")
    else:
        st.write("ECHEC du préprocessing des duplicates")

    # Count duplicate timestamps
    duplicate_timestamps = raw_train_g[raw_train_g.duplicated(subset='Time', keep=False)]

    # Display the count of duplicate timestamps
    num_duplicates = duplicate_timestamps.shape[0]
    st.write(f"Number of lines with duplicate timestamps: {num_duplicates}")

    if (num_duplicates==0):
        st.write("Plus aucun duplicate à présent : we can preprocess the lines with diff more than one seconds")
    else:
        st.write("There are still duplicate data")

    # Préprocessing -  Constance de la périodicité

    #Check that we still have the same lines with more than one seconds

    # Filter rows where the time difference is greater than 1 second
    long_diff_g = raw_train_g[raw_train_g['Time_diff'] > pd.Timedelta(seconds=1)]

    # Create a table with the two times and their differences
    table_g = long_diff_g[['Time', 'Time_diff']]

    # Ensure both DataFrames have the same indices and fill NaT values if needed
    table_g = table_g.reindex_like(table).fillna(table)

    # Display the table
    st.write(table_g)
    st.write(table)
    st.write(table==table_g)

    # Set the 'Time' column as the index
    raw_train_g.set_index('Time', inplace=True)

    # Generate a complete time range from the start to the end with a frequency of 1 second
    full_time_index = pd.date_range(start=raw_train_g.index.min(), end=raw_train_g.index.max(), freq='1s')

    # Reindex the dataframe to have 1 second intervals, filling in missing values using forward fill
    raw_train_resampled = raw_train_g.reindex(full_time_index, method='ffill')

    # Reset the index if you want the 'Time' column back as a column instead of the index
    raw_train_resampled.reset_index(inplace=True)
    raw_train_resampled.rename(columns={'index': 'Time'}, inplace=True)

    # Display the first few rows of the transformed dataset
    st.write(raw_train_resampled.head())

    #Check the resampling

    # Find the index of the row with Time = '2024-06-10 22:00:00'
    target_time = pd.Timestamp('2024-06-10 22:00:00')

    # Find the index of the target time
    target_index = raw_train_resampled[raw_train_resampled['Time'] == target_time].index[0]

    # Get 5 rows before and 1 after the target row
    start_index = max(target_index - 5, 0)  # Make sure the index is not negative
    end_index = target_index + 2  # 1 row after + 1 for the target row itself

    # Display the rows
    subset = raw_train_resampled.iloc[start_index:end_index]
    st.write(subset)

    # Update the time difference between consecutive rows
    raw_train_resampled['Time_diff'] = raw_train_resampled['Time'].diff()        

    #Check the results
    # Filter rows where the time difference is greater than 1 second
    long_diff_g = raw_train_resampled[raw_train_resampled['Time_diff'] > pd.Timedelta(seconds=1)]

    # Create a table with the two times and their differences
    table_g = long_diff_g[['Time', 'Time_diff']]

    # Display the table
    st.write(table_g)

    # Sum the time where the time difference is not 1 second
    non_constant_frequency = raw_train_resampled['Time_diff'].iloc[1:].ne(pd.Timedelta(seconds=1)).sum()
    # Count the number of cases where the time difference is not 1 second
    num_rows_long_diff = len(raw_train_resampled[raw_train_resampled['Time_diff'] > pd.Timedelta(seconds=1)])


    st.write(f"\n********** RECAPITULATIF DES VALIDATIONS ET PRETRAITEMENTS DES DONNEES **********")
    st.write(f"Number of lines with duplicate timestamps: {num_duplicates}")
    st.write(f"Sum of time where the frequency is not constant: {non_constant_frequency}")
    st.write(f"Number of rows with time difference not equal to 1 second: {num_rows_long_diff}")

    # Préprocessing - Split entre données d'entrainement et de test

    orig_split_factor = 2080027/2398267 #répartition train/test utilisé lors de la phase 1 (2398267 = 2080027 + 318240)

    # Split the data
    #donnees_raw_train, donnees_raw_valid = train_test_split(raw_train_resampled, train_size=orig_split_factor, random_state=42, shuffle=False)
    #Split de la phase 1 de l'Hackathon
    donnees_raw_train, donnees_raw_valid = train_test_split(input_data, train_size=orig_split_factor, random_state=42, shuffle=False)

    # Display the sizes
    st.write(f"Taille des données d'entrainement: {len(donnees_raw_train)}")
    st.write(f"Taille des données de test: {len(donnees_raw_valid)}")

    #donnees_raw_train.iloc[:1] = donnees_raw_train.iloc[:1].applymap(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if isinstance(x, pd.Timestamp) else x)
    st.write("Première ligne du jeu de données d'entrainement:", donnees_raw_train.iloc[:1])
    st.write("Première ligne du jeu de données de test:", donnees_raw_valid.iloc[:1])

    return donnees_raw_train, donnees_raw_valid

# Définir la fonction de Preprocessing

def preprocesser_les_donnees_2 (preprocessing_dir, donnees_raw, window_size_x, window_size_y, step, subset):

    # Extract the Bits/s column
    bits_per_second = donnees_raw["Bits/s"].values

    # Prepare x_subset and y_subset
    x_subset = []
    y_subset = []

    # Iterate over the data using the sliding window
    for start_idx in range(0, len(bits_per_second) - window_size_x - window_size_y + 1, step):
        # Extract x_subset (window_size_x seconds)
        x_segment = bits_per_second[start_idx : start_idx + window_size_x]
        # Extract y_subset (next window_size_y seconds)
        y_segment = bits_per_second[start_idx + window_size_x : start_idx + window_size_x + window_size_y]
        
        # Append to the lists
        x_subset.append(x_segment)
        y_subset.append(y_segment)

    # Convert to NumPy arrays
    x_subset = np.array(x_subset)
    y_subset = np.array(y_subset)

    # Save to CSV files
    #pd.DataFrame(x_train).to_csv("x_"+subset+".csv", index=False, header=False)
    #pd.DataFrame(y_train).to_csv("y_"+subset+".csv", index=False, header=False)
    pd.DataFrame(x_subset).to_csv(preprocessing_dir+"x_"+subset+"_s"+str(step)+"_o"+str(window_size_x)+"_p"+str(window_size_y)+".csv", index=False, header=False)
    pd.DataFrame(y_subset).to_csv(preprocessing_dir+"y_"+subset+"_s"+str(step)+"_o"+str(window_size_x)+"_p"+str(window_size_y)+".csv", index=False, header=False)

    print("x_"+subset+"_s"+str(step)+"_o"+str(window_size_x)+"_p"+str(window_size_y)+".csv and y_"+subset+"_s"+str(step)+"_o"+str(window_size_x)+"_p"+str(window_size_y)+".csv saved successfully.")
    print(f"x_{subset}_s{step}_o{window_size_x}_o{window_size_y} shape: {x_subset.shape}, y_{subset}_s{step}_o{window_size_x}_p{window_size_y} shape: {y_subset.shape}\n")

    return 1

