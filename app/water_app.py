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

# Load example data from CSV (The averages of each GRU)
GRU_averages = pd.read_csv('assets/GRU_averages.csv')
resource_units = GRU_averages['resource_unit']
GRU_averages = GRU_averages.drop(['resource_unit'],axis=1)
example_values = GRU_averages.iloc[0].to_dict()


# Define slider labels
slider_labels = [
    #("DMS (mg/L)", "DMS (mg/L)", 0.01, 5000.00),
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

#create example data and list the resource units by name, note the resource units only contains the features of the units.
selected_example = st.sidebar.selectbox("Select Example of averaged Resource unit water properties", ["Custom"] + list(resource_units) )

# Load the GRU image
GRU_image = "assets/LowerOrangeCatchments_GRU.jpg"

# Create a container for the image on the right side
col2 = st.columns([1, 1])

# Display the image in the right column
with col2[1]:
    st.image(GRU_image, caption="Map of Lower Orange river with respected Ground resource units")

st.title("Water sample predictor")
st.write("This app uses chemical substances of a water sample to predict  various elements of interest for the user using machine learning models,"
         " following are the models with their respected accuracies and what they predicted:"
              )
st.markdown(
    """
    1. Random forest (99.99% accuracy) to dermine if the water is **safe for consumption.**
    2. Random forest (99.99% accuracy) to predict the **water class**, class 0 is the best class 4 is the worst.  
    3. Linear regression (99.5% accuracy) to predict the total amount of milligrams per Liter of **Disolved Mineral Salts (DMS)** or also called **Total Disolved Solids (TDS)**  
    4. Deep Neural Network (66.7% accuracy) to predict from what **Resource Unit** the water came from.
    """
)

# receive the input data so that it can be used for prediction
input_data = {}

st.sidebar.title("Water Sample Input")

for label, key, min_value, max_value in slider_labels:
    if selected_example != "Custom":
        initial_value = GRU_averages.loc[list(resource_units).index(selected_example)][key]
    else:
        initial_value = 0.01  # Default to zero for custom input
    input_data[key] = st.sidebar.slider(label, min_value=min_value, max_value=max_value, key=key, value=initial_value)


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
