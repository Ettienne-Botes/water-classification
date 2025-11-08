import streamlit as st
import pickle
import pandas as pd

# --- Data Loading and Initial Setup ---
# (Assuming 'model' and 'assets' directories exist with files)

# Load the models and mappings
try:
    with open('model/waterClass_model.pkl', 'rb') as f:
        waterClass_model = pickle.load(f)
    with open('model/DMS_model.pkl', 'rb') as f:
        DMS_model = pickle.load(f)
    with open('model/model_dnn.pkl', 'rb') as f:
        DNN_model = pickle.load(f)
    with open('model/label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Missing model file: {e}. Please ensure the 'model/' directory contains all necessary files.")
    st.stop() # Stop the app if models can't load

# Load example data from CSV
try:
    GRU_averages = pd.read_csv('assets/GRU_averages.csv')
except FileNotFoundError:
    st.error("Missing asset file: assets/GRU_averages.csv. Please check the path.")
    st.stop()

resource_units = GRU_averages['resource_unit']
GRU_data = GRU_averages.drop(['resource_unit'], axis=1)

# Create a mapping for easy lookup (key = resource_unit name, value = row of data)
# This is a cleaner way to get the data later
GRU_map = GRU_data.set_index(resource_units)


# Define slider labels
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

# --- Session State Management Functions ---

def update_slider_values():
    """Callback function to update all slider values in session state
    when the selectbox changes. This is the crucial fix!
    """
    selected = st.session_state.select_example_box
    
    if selected != "Custom":
        # Look up the row of data from the GRU_map
        new_values = GRU_map.loc[selected].to_dict()
        
        # Update session state for each slider key
        for _, key, _, _ in slider_labels:
            st.session_state[key] = new_values[key]
    else:
        # Reset to default for "Custom" input
        for _, key, _, _ in slider_labels:
            st.session_state[key] = 0.01


# --- Session State Initialization ---

st.sidebar.title("Water Sample Input")

# Initialize session state for the selectbox and all input variables
if 'select_example_box' not in st.session_state:
    st.session_state.select_example_box = 'Custom'

for _, key, _, _ in slider_labels:
    if key not in st.session_state:
        # Initialize with the custom default value
        st.session_state[key] = 0.01


# 1. Selectbox: Use 'on_change' to trigger the update function
selected_example = st.sidebar.selectbox(
    "Select Example of averaged Resource unit water properties",
    ["Custom"] + list(resource_units),
    key='select_example_box',
    on_change=update_slider_values # <--- This fires when a new example is chosen
)

# 2. Sliders: The 'value' argument now must reference st.session_state[key]
input_data = {}
for label, key, min_value, max_value in slider_labels:
    # When the script re-runs, the 'value' is taken from the session state, 
    # which was just updated by update_slider_values() if the selectbox changed.
    # The slider's manual input also automatically updates st.session_state[key].
    st.sidebar.slider(
        label, 
        min_value=min_value, 
        max_value=max_value, 
        key=key, 
        value=st.session_state[key]
    )
    # The input data for prediction is simply the current state of the sliders
    input_data[key] = st.session_state[key]


# --- Main App Content ---

# Load the GRU image
GRU_image = "assets/LowerOrangeCatchments_GRU.jpg"

st.title("Water sample predictor")

# Create a container for the image on the right side
col1, col2 = st.columns([2, 1])

# Display the image in the right column
with col2:
    st.image(GRU_image, caption="Map of Lower Orange river with respected Ground resource units") 

with col1:
    st.write("This app uses chemical substances of a water sample to predict various elements of interest for the user using machine learning models. Following are the models with their respected accuracies and what they predicted:")
    st.markdown(
        """
        1. Random forest (**99.99% accuracy**) to dermine if the water is **safe for consumption.**
        2. Random forest (**99.99% accuracy**) to predict the **water class**, class 0 is the best class 4 is the worst.  
        3. Linear regression (**99.5% accuracy**) to predict the total amount of milligrams per Liter of **Disolved Mineral Salts (DMS)** or also called **Total Disolved Solids (TDS)** 4. Deep Neural Network (**66.7% accuracy**) to predict from what **Resource Unit** the water came from.
        """
    )


# Predict water safety
if st.button("Predict"):
    # The input data is ready from st.session_state!
    input_df = pd.DataFrame(input_data, index=[0])
    
    # 1. Water Class/Safety
    waterClass_prediction = waterClass_model.predict(input_df)
    waterClass_prediction = round(waterClass_prediction[0])
    water_safety = 'Yes'
    if (waterClass_prediction == 3 or waterClass_prediction == 4):
        water_safety = 'No'
    
    # 2. DMS
    DMS_prediction = DMS_model.predict(input_df).round(2)
    
    # 3. DNN/GRU Prediction
    DNN_prediction = DNN_model.predict(input_df)
    predicted_index = DNN_prediction.argmax()
    predicted_label = label_mapping[predicted_index]

    predicted_gru_names = [label_mapping[i] for i in range(len(DNN_prediction[0]))]
    
    # Round the probabilities to 2 decimal places and convert to percentages
    rounded_probs_percentage = [round(prob * 100, 2) for prob in DNN_prediction[0]]

    # Create a list of tuples containing GRU names and their rounded probabilities as percentages
    data_GRU = [("Resource unit", "Percentage predicted")] +[(gru_name, 
    f"{prob}%") for gru_name, prob in zip(predicted_gru_names, rounded_probs_percentage)]

    st.header("Results")
    st.markdown(
        f"Is the water safe for consumption?: **:{'green' if water_safety == 'Yes' else 'red'}[{water_safety}]**"
    )
    st.markdown(f"Water class: **:blue[{waterClass_prediction}]**")
    st.markdown(f"Prediction of Total mg/L of Disolved Mineral Salts in water: **:blue[{DMS_prediction[0]} mg/L]**")
    st.markdown(f"Ground Resource Unit predicted: **:blue[{predicted_label}]**")

    # Display the predicted GRU names and their corresponding probabilities in a table
    st.write("Ground resource unit probability distribution:")
    st.table(data_GRU)
