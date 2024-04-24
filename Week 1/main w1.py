import streamlit as st
import numpy as np
import pandas as pd

def learn(concepts, target):
    '''
    learn() function implements the learning method of the Candidate elimination algorithm.
    Arguments:
        concepts - a data frame with all the features
        target - a data frame with corresponding output values
    '''
    # Initialize S0 with the first instance from concepts
    specific_h = concepts[0].copy()

    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    # The learning iterations
    for i, h in enumerate(concepts):
        # Checking if the hypothesis has a positive target
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                # Change values in S & G only if values change
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        # Checking if the hypothesis has a positive target
        if target[i] == "No":
            for x in range(len(specific_h)):
                # For negative hypothesis change values only in G
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    # find indices where we have empty rows, meaning those that are unchanged
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        # remove those rows from general_h
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    # Return final values
    return specific_h, general_h

def main():
    st.title("Candidate Elimination Algorithm")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Reading uploaded CSV data
            data = pd.read_csv(uploaded_file)
            st.write("Original Data:")
            st.write(data)

            # Separating concept features from Target
            concepts = np.array(data.iloc[:, :-1])

            # Isolating target into a separate DataFrame
            target = np.array(data.iloc[:, -1])

            s_final, g_final = learn(concepts, target)
            st.write("\nFinal Specific_h:")
            st.write(s_final)
            st.write("\nFinal General_h:")
            st.write(g_final)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
