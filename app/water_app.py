import streamlit as st
import pickle
import pandas as pd

# Load the water classification model
with open('model/waterClass_model.pkl', 'rb') as f:
    waterClass_model = pickle.load(f)

#Load the DMS model
with open('model/DMS_model.pkl', 'rb') as f:
    DMS_model = pickle.load(f)

#Load the Deep Neural network model
with open('model/model_dnn.pkl', 'rb') as f:
    DNN_model = pickle.load(f)

#label mapping to convert the GRU's from a list of intergers to the list of GRU names
with open('model/label_mapping.pkl', 'rb') as f:
    label_mapping = pickle.load(f)

GRU_averages = pd.read_csv('assets/GRU_averages.csv')
resource_units = GRU_averages['resource_unit']
GRU_averages = GRU_averages.drop(['resource_unit'],axis=1)
GRU_map = GRU_averages.set_index(resource_units)


slider_labels = [
    ("EC-(mS/m)", "EC-(mS/m)", 0.01, 1000.00),
    ("pH-Diss-Water (PH)", "pH-Diss-Water (PH)", 0.01, 12.00),
    ("Ca (mg/L)", "Ca (mg/L)", 0.01, 500.00),
    ("Mg (mg/L)", "Mg (mg/L)", 0.01, 1000.00),
    ("Na (mg/L)", "Na (mg/L)", 0.01, 2000.00),
    ("K (mg/L)", "K (mg/L)", 0.01, 1000.00),
    ("TAL CaCO3 (mg/L)", "TAL CaCO3 (mg/L)", 0.01, 1000.00),
    ("Cl (mg/L)", "Cl (mg/L)", 0.01, 2000.00),
    ("SO4 (mg/L)", "SO4 (mg/L)", 0.01, 2000.00),
    ("F (mg/L)", "F (mg/L)", 0.01, 5.00)
]


if 'selected_example' not in st.session_state:
    st.session_state.selected_example = 'Custom'

for _, key, _, _ in slider_labels:
    if key not in st.session_state:
        st.session_state[key] = 0.01 

def update_session_state_from_example():
    """Callback function to update slider values when the selectbox changes."""
    selected = st.session_state.select_example_box
    if selected != "Custom":
        new_values = GRU_map.loc[selected].to_dict()
        for _, key, _, _ in slider_labels:
            st.session_state[key] = new_values[key]
    else:

        for _, key, _, _ in slider_labels:
            if key in st.session_state:
                st.session_state[key] = 0.01


st.sidebar.title("Water Sample Input")

selected_example = st.sidebar.selectbox(
    "Select Example of averaged Resource unit water properties",
    ["Custom"] + list(resource_units),
    key='select_example_box',
    on_change=update_session_state_from_example
)


input_data = {}
for label, key, min_value, max_value in slider_labels:
    input_data[key] = st.sidebar.slider(
        label, 
        min_value=min_value, 
        max_value=max_value, 
        key=key, 
        value=st.session_state[key] 
    )


# Predict water safety
if st.button("Predict"):
    input_df = pd.DataFrame(input_data, index=[0])
    waterClass_prediction = waterClass_model.predict(input_df)
    waterClass_prediction = round(waterClass_prediction[0])
    water_safety = 'Yes'
    if (waterClass_prediction==3 or waterClass_prediction==4):
        water_safety = 'No'
    DMS_prediction = DMS_model.predict(input_df).round(2)
    DNN_prediction = DNN_model.predict(input_df)
    predicted_index = DNN_prediction.argmax()
    predicted_label = label_mapping[predicted_index]

    predicted_gru_names = [label_mapping[i] for i in range(len(DNN_prediction[0]))]
    
    # Round the probabilities to 4 decimal places and convert to percentages
    rounded_probs_percentage = [round(prob * 100, 2) for prob in DNN_prediction[0]]

    # Create a list of tuples containing GRU names and their rounded probabilities as percentages
    data_GRU = [("Resource unit", "Percentage predicted")] +[(gru_name, 
    f"{prob}%") for gru_name, prob in zip(predicted_gru_names, rounded_probs_percentage)]

    st.markdown("Is the water safe for consumption?: **:green[Yes]**" if water_safety == "Yes" else 
    "Is the water safe for consumption?: **:red[No]**")
    st.markdown(f"Water class: **:blue[{waterClass_prediction}]**")
    st.markdown(f"Prediction of Total mg/L of Disolved Mineral Salts in water: **:blue{DMS_prediction}**")
    st.markdown(f"Ground Resource Unit predicted: **:blue[{predicted_label}]**")

    # Display the predicted GRU names and their corresponding probabilities in a table
    st.write("Ground resource unit probability distribution:")
    st.table(data_GRU)
