# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

import random

st.title( 'Airline Customer Satisfaction') 
st.write('**Gain insights into passenger experiences and improe satisfaction through' \
'data analysis and surveys.**')
st.image('airline.jpg', width=600)
st.markdown(
    "<p style='color: grey; font-size: 12px;'>Understand your customers to improve your airline services!</p>",
    unsafe_allow_html=True
)
with st.expander("**What can you do with this app?**"):
    st.markdown("""
    - 📝 **Fill out a Survey:** Provide a form for users to fill out their airline satisfaction feedback.
    - 📊 **Make Data-Driven Decisions:** Use insights to guide improvements in customer experience.
    - wrench **Interactive Features:** Explore data with fully interactive charts and summaries!
    """)

st.title("Prediction of Customer Satisfaction" \
" (Decision Tree)")
st.info("ℹ️ Please fill out the survey form "
"in the sidebar and click **Predict** to see the satisfaction prediction.")

# Load the pre-trained model from the pickle file
dt_airline = open('dt_airline.pickle', 'rb') 
clf = pickle.load(dt_airline) 
dt_airline.close()






# features = airline_df[['customer_type', 'type_of_travel', 'class', 
#                        'age', 'flight_distance', 'departure_delay_in_minutes',
#                        'arrival_delay_in_minutes','seat_comfort','food_and_drink'
#                        ,'inflight_wifi_service',
#                        'inflight_entertainment','online_support',
#                        'ease_of_online_booking','on-board_service',
#                        'leg_room_service','baggage_handling',
#                        'checkin_service','cleanliness','online_boarding',
#                        'departure_arrival_time_convenient'           
#  ]]

#---------------------------------------------------------------------------------------------
# Using Default (Original) Dataset to Automate Few Items
#---------------------------------------------------------------------------------------------

# Load the default dataset
default_df = pd.read_csv('airline.csv')
# default_df = default_df.dropna().reset_index(drop = True) 
default_df.dropna(inplace = True)
# NOTE: drop = True is used to avoid adding a new column for old index
noprice =  default_df.drop(columns = ['satisfaction'])



st.sidebar.header('Airline Customer Satisfaction Survey')

st.sidebar.subheader('Part 1: Customer Details')
st.sidebar.write('Provide Information about the customer flying')

# cust_type_ORDER = ["Loyal Customer", "Disloyal Customer"]
# travel_type_ORDER = ["Personal Travel", "Business Travel"]
# class_ORDER = ["Eco", "Eco Plus", "Business"]

customer_type = st.sidebar.selectbox("What type of customer is this?", default_df['customer_type'].unique())
type_of_travel = st.sidebar.selectbox("Is the customer travelling for business or personal reasons?", default_df['type_of_travel'].unique())
class_type = st.sidebar.selectbox(      # ✅ use a new variable
    "In which class will the customer be flying?",
    default_df['class'].unique()
)
age = st.sidebar.number_input(
    'How old is the customer?',
    min_value=int(default_df['age'].min()),
    max_value=int(default_df['age'].max()),
    step=1
)

st.sidebar.subheader('Part 2: Flight Details')
st.sidebar.write("Provide details about the customer's flight details")

miles = st.sidebar.number_input('How far is the customer flying in miles?',
    min_value=int(default_df['flight_distance'].min()),
    max_value=int(default_df['flight_distance'].max()),
    step=1
)
departure_delay_in_minutes = st.sidebar.number_input('How many minutes was the customer’s departure delayed? (Enter 0 if not delayed)',
    min_value=int(default_df['departure_delay_in_minutes'].min()),
    max_value=int(default_df['departure_delay_in_minutes'].max()),
    step=1
)
arrival_delay_in_minutes = st.sidebar.number_input('How many minutes was the customer’s arrival delayed? (Enter 0 if not delayed)',
    min_value=int(default_df['arrival_delay_in_minutes'].min()),
    max_value=int(default_df['arrival_delay_in_minutes'].max()),
    step=1
)

st.sidebar.subheader('Part 3: Customer Experience Details')
st.sidebar.write("Provide details about the customer's flight experience and satisfaction")

seat_comfort = st.sidebar.radio("Seat comfort (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
food_and_drink = st.sidebar.radio("Food and drink service (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
inflight_wifi_service = st.sidebar.radio("Inflight WiFi service (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
inflight_entertainment = st.sidebar.radio("Inflight entertainment (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
gate_location = st.sidebar.radio("Rate the gate lcoation (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
online_support = st.sidebar.radio("Online support (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
ease_of_online_booking = st.sidebar.radio("Ease of online booking (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
on_board_service = st.sidebar.radio("On-board service (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
leg_room_service = st.sidebar.radio("Leg room comfort (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
baggage_handling = st.sidebar.radio("Baggage handling efficiency (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
checkin_service = st.sidebar.radio("Check-in service (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
cleanliness = st.sidebar.radio("Aircraft cleanliness (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
online_boarding = st.sidebar.radio("Online boarding (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)
departure_arrival_time_convenient = st.sidebar.radio("Departure/arrival time convenience (1–5 stars)", [1, 2, 3, 4, 5], horizontal=True)

# ✅ Button at the bottom (no form)
predict_button = st.sidebar.button("✈️ Predict Satisfaction")
if predict_button:
    st.success("Prediction executed successfully!")

    # Copy base dataframe (no target column)
    encode_df = default_df.drop(columns=['satisfaction']).copy()

    # --- Append user input row in the same order as model training ---
    encode_df.loc[len(encode_df)] = [
        customer_type,                  # customer_type
        type_of_travel,                 # type_of_travel
        class_type,                     # class
        age,                            # age
        miles,                          # flight_distance
        departure_delay_in_minutes,     # departure_delay_in_minutes
        arrival_delay_in_minutes,       # arrival_delay_in_minutes
        seat_comfort,                   # seat_comfort
        food_and_drink,                 # food_and_drink
        gate_location,                  # gate_location
        inflight_wifi_service,          # inflight_wifi_service
        inflight_entertainment,         # inflight_entertainment
        online_support,                 # online_support
        ease_of_online_booking,         # ease_of_online_booking
        on_board_service,               # on-board_service
        leg_room_service,               # leg_room_service
        baggage_handling,               # baggage_handling
        checkin_service,                # checkin_service
        cleanliness,                    # cleanliness
        online_boarding,                # online_boarding
        departure_arrival_time_convenient  # departure_arrival_time_convenient
    ]

    # --- Normalize categorical text for exact matching ---
    for col in ['customer_type', 'type_of_travel', 'class']:
        encode_df[col] = encode_df[col].astype(str).str.strip().str.title()

    # --- One-hot encode only categorical columns ---
    encode_dummy_df = pd.get_dummies(encode_df, columns=['customer_type', 'type_of_travel', 'class'])

    # --- Extract just the user’s row ---
    user_encoded_df = encode_dummy_df.tail(1)

# --- Align columns with model training order ---
    user_encoded_df = user_encoded_df.reindex(columns=clf.feature_names_in_, fill_value=0)

# --- Explicitly convert every column to float ---
    for col in user_encoded_df.columns:
        try:
            user_encoded_df[col] = user_encoded_df[col].astype(float)
        except Exception:
            # if somehow a value cannot be converted, fill with 0
            user_encoded_df[col] = pd.to_numeric(user_encoded_df[col], errors='coerce').fillna(0)


    # --- Sanity check before prediction ---
    non_numeric = user_encoded_df.select_dtypes(exclude=['number']).columns
    if len(non_numeric) > 0:
        st.error(f"Non-numeric columns remain: {list(non_numeric)}")
        st.stop()

    # ✅ --- Now predict safely ---
    try:
        predicted_satisfaction = clf.predict(user_encoded_df)[0]
        predicted_proba = clf.predict_proba(user_encoded_df)[0]
        confidence = max(predicted_proba) * 100

        # --- Format the prediction nicely ---
        # Define color depending on satisfaction
        color = "green" if predicted_satisfaction.lower() == "satisfied" else "red"

        # --- Display result in a styled container ---
        st.markdown(
            f"""
            <div style="
                background-color:#ffffff;
                border:2px solid #e0e0e0;
                border-radius:10px;
                padding:20px;
                margin-top:20px;
                text-align:center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            ">
                <h3 style="color:#333333;">Prediction Result</h3>
                <p style="font-size:18px; color:{color}; font-weight:bold;">
                    Your predicted satisfaction level is {predicted_satisfaction}.
                </p>
                <p style="font-size:16px; color:#555555;">
                    With a confidence of {confidence:.2f}%.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("DEBUG: user_encoded_df dtypes:")
        st.write(user_encoded_df.dtypes)
        st.write("DEBUG: user_encoded_df sample row:")
        st.write(user_encoded_df.head())


# # When the Predict button is clicked
# if predict_button:
#     st.success("Prediction executed successfully!")

#     # Copy base dataframe (no target column)
#     encode_df = default_df.drop(columns=['satisfaction']).copy()

#     # ✅ Add the user's input row to the dataframe
#     encode_df.loc[len(encode_df)] = [
#     customer_type,                  # 1. customer_type
#     type_of_travel,                 # 2. type_of_travel
#     class_type,                     # 3. class
#     age,                            # 4. age
#     miles,                          # 5. flight_distance
#     departure_delay_in_minutes,     # 6. departure_delay_in_minutes
#     arrival_delay_in_minutes,       # 7. arrival_delay_in_minutes
#     seat_comfort,                   # 8. seat_comfort
#     food_and_drink,                 # 9. food_and_drink
#     gate_location,                  # 10. gate_location
#     inflight_wifi_service,          # 11. inflight_wifi_service
#     inflight_entertainment,         # 12. inflight_entertainment
#     online_support,                 # 13. online_support
#     ease_of_online_booking,         # 14. ease_of_online_booking
#     on_board_service,               # 15. on-board_service
#     leg_room_service,               # 16. leg_room_service
#     baggage_handling,               # 17. baggage_handling
#     checkin_service,                # 18. checkin_service
#     cleanliness,                    # 19. cleanliness
#     online_boarding,                # 20. online_boarding
#     departure_arrival_time_convenient  # 21. departure_arrival_time_convenient
# ]

# #
#     # One-hot encode categorical variables
#     #encode_dummy_df = pd.get_dummies(encode_df)
#     # Ensure all categorical columns are strings
#     # encode_df['customer_type'] = encode_df['customer_type'].astype(str)
#     # encode_df['type_of_travel'] = encode_df['type_of_travel'].astype(str)
#     # encode_df['class'] = encode_df['class'].astype(str)

# # One-hot encode only categorical variables
#     encode_dummy_df = pd.get_dummies(encode_df, columns=['customer_type','type_of_travel','class'])
#     user_encoded_df = encode_dummy_df.tail(1)
#     user_encoded_df = user_encoded_df.reindex(columns=clf.feature_names_in_, fill_value=0)


  
#     # 🧠 Predict satisfaction using your trained model
#     predicted_satisfaction = clf.predict(user_encoded_df)[0]
#     predicted_proba = clf.predict_proba(user_encoded_df)[0]
#     confidence = max(predicted_proba) * 100

#     # 🎯 Show prediction header
#     st.markdown("<h2 style='color: green;'>Prediction Complete</h2>", unsafe_allow_html=True)

#     # Show the predicted satisfaction result
#     st.markdown(f"<h3>Predicted Satisfaction</h3>", unsafe_allow_html=True)
#     st.markdown(
#         f"<h1 style='font-weight: bold; color:#0066cc;'>{predicted_satisfaction}</h1>",
#         unsafe_allow_html=True
#     )

#     # 📊 Display prediction confidence
#     st.markdown(
#         f"""
#         <div style="
#             background-color:#e6f0ff;
#             padding:10px;
#             border-radius:8px;
#             border:1px solid #99c2ff;
#             width:fit-content;
#         ">
#             <b>Prediction Confidence:</b> {confidence:.2f}%
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

with st.expander("🧍 Customer Type Comparison"):
    pct = (default_df['customer_type'] == customer_type).mean() * 100
    st.write(f"**Your selection:** {customer_type}")
    st.write(f"Percentage of our flyers with this selection: {pct:.1f}%")

with st.expander("🧳 Type of Travel Comparison"):
    pct = (default_df['type_of_travel'] == type_of_travel).mean() * 100
    st.write(f"**Your selection:** {type_of_travel}")
    st.write(f"Percentage of our flyers with this selection: {pct:.1f}%")

with st.expander("💺 Flight Class Comparison"):
    pct = (default_df['class']==class_type).mean() * 100
    st.write(f"**Your selection:** {class_type}")
    st.write(f"Percentage of our flyers with this selection: {pct:.1f}%")

with st.expander("🎂 Age Group Comparison"):
    # Example: group by age range
    bins = [0, 18, 30, 45, 60, 75, 100]
    labels = ['Under 18', '18–30', '31–45', '46–60', '61–75', '76+']
    default_df['age_group'] = pd.cut(default_df['age'], bins=bins, labels=labels, right=False)
    
    # Compute user's age group
    user_age_group = pd.cut([age], bins=bins, labels=labels, right=False)[0]
    pct = (default_df['age_group'] == user_age_group).mean() * 100

    st.write(f"**Your age group:** {user_age_group}")
    st.write(f"Percentage of our flyers in this group: {pct:.1f}%")
