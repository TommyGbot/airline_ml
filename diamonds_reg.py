# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

import random

# List of diamond fun facts
diamond_facts = [
    "💎 The largest diamond ever discovered is the Cullinan Diamond, weighing over 3,100 carats!",
    "🔍 Diamonds are made of pure carbon—the same element found in graphite pencils!",
    "🔥 Natural diamonds can form more than 100 miles beneath Earth's surface under extreme heat and pressure.",
    "🪨 Only about 20% of mined diamonds are suitable for jewelry—the rest are used for industrial purposes.",
    "🌈 The word 'diamond' comes from the Greek word 'adamas', meaning 'unbreakable' or 'invincible'.",
    "🕵️‍♂️ Lab-grown diamonds have the same physical, chemical, and optical properties as natural ones!",
    "💰 The price of a diamond increases exponentially with its carat weight—not linearly!",
    "💡 Most diamonds are between 1 and 3 billion years old—older than most continents!",
    "🎨 The rarest diamond color is red—only a handful have ever been found.",
    "🪞 The hardest known natural material on Earth is diamond—it ranks 10 on the Mohs hardness scale."
]

st.title('💎 Diamond Price Prediction') 




# Display the image
st.write("This app uses multiple inputs to predict" \
" the price of a diamond") 
st.image('diamond_image.jpg', width = 600)


st.info("""
    Please choose a data input method to proceed
    """)
alpha = st.slider("**Select alpha value (confidence level)**", min_value=0.01, max_value=0.30, value=0.10, step=0.01,
                          help="e.g., 0.10 = 90% prediction interval")
alpha_percent = 1-alpha
# Load the pre-trained model from the pickle file
reg_diamond = open('reg_diamond.pickle', 'rb') 
clf = pickle.load(reg_diamond) 
reg_diamond.close()

# Create a sidebar for input collection
st.sidebar.header('Diamond Features Input')
st.sidebar.write('You can either upload your data file or input the features manually')



#---------------------------------------------------------------------------------------------
# Using Default (Original) Dataset to Automate Few Items
#---------------------------------------------------------------------------------------------

# Load the default dataset
default_df = pd.read_csv('diamonds.csv')
default_df = default_df.dropna().reset_index(drop = True) 
# NOTE: drop = True is used to avoid adding a new column for old index
noprice =  default_df.drop(columns = ['price'])

with st.sidebar.expander("Option 1: 📂 CSV Upload"):
    diamond_file = st.file_uploader("Upload a diamonds CSV", type=["csv"])
    st.caption("Expected columns: carat, cut, color, clarity, depth, table, x, y, z")
    st.dataframe(noprice.head(5), use_container_width=True)

with st.sidebar.expander("Option 2: 📝 Manual Input (Form)"):
    with st.form("diamond_input_form"):  # <— create a form with a unique key
        st.write("Enter diamond features below:")

        carat = st.number_input(
            'Carat Weight',
            min_value=float(default_df['carat'].min()),
            max_value=float(default_df['carat'].max()),
            step=0.01,
            help="Weight of the diamond in carats"
        )

        CUT_ORDER    = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
        COLOR_ORDER  = ["J", "I", "H", "G", "F", "E", "D"]
        CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

        cut = st.selectbox("Cut", CUT_ORDER)
        color = st.selectbox("Color", COLOR_ORDER)
        clarity = st.selectbox("Clarity", CLARITY_ORDER)

        x = st.number_input(
            'Length (x in mm)',
            min_value=float(default_df['x'].min()),
            max_value=float(default_df['x'].max()),
            step=0.01
        )
        y = st.number_input(
            'Width (y in mm)',
            min_value=float(default_df['y'].min()),
            max_value=float(default_df['y'].max()),
            step=0.01
        )
        z = st.number_input(
            'Depth (z in mm)',
            min_value=float(default_df['z'].min()),
            max_value=float(default_df['z'].max()),
            step=0.01
        )

        depth = z / ((x + y) / 2) if x > 0 and y > 0 else 0

        table = st.number_input(
            'Table (%)',
            min_value=float(default_df['table'].min()),
            max_value=float(default_df['table'].max()),
            value=float(default_df['table'].mean()),
            step=0.1
        )

        # 🔘 Add the submit button here
        submitted = st.form_submit_button("✅ Submit Form Data")

# with st.sidebar.expander("Option 2: 📝 Manual Input (Form)"):
#     carat = st.number_input('carat', 
#                                     min_value = default_df['carat'].min(), 
#                                     max_value = default_df['carat'].max(), 
#                                     step = .01,
#                                     help="Weight of the diamond in carats")


#     # For categorical variables, using selectbox
#     CUT_ORDER    = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
#     COLOR_ORDER  = ["J", "I", "H", "G", "F", "E", "D"]
#     CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

#     cut = st.selectbox("cut", CUT_ORDER, help="Cut quality rating") #asked chat GPT to help with order of inputs
#     color = st.selectbox("color", COLOR_ORDER, help="Color grade (J worst → D best)")
#     clarity = st.selectbox("clarity", CLARITY_ORDER, help="Clarity grade (I1 worst → IF best)")

#     x = st.number_input('Length in mm', 
#                                     min_value = default_df['x'].min(), 
#                                     max_value = default_df['x'].max(), 
#                                     step = .01,
#                                     help="Length of the diamond in millimeters")

#     y = st.number_input('Width in mm', 
#                                     min_value = default_df['y'].min(), 
#                                     max_value = default_df['y'].max(), 
#                                     step = .01,
#                                     help="Width of the diamond in millimeters")

#     z = st.number_input('Depth in mm', 
#                                     min_value = default_df['z'].min(), 
#                                     max_value = default_df['z'].max(), 
#                                     step = .01,
#                                     help="Depth of the diamond in millimeters")

#     depth = z / np.mean([x, y])

#     table = st.number_input(
#         'table',
#         min_value=float(default_df['table'].min()),
#         max_value=float(default_df['table'].max()),
#         value=float(default_df['table'].mean()),   # ✅ make sure this is numeric, not string
#         step=1.0                                   # ✅ match type with float arguments
#     )


with st.sidebar.expander("💡 Did You Know?"):
    st.write(random.choice(diamond_facts))


# If no file is provided, then allow user to provide inputs using the form
if diamond_file is None:
        # Encode the inputs for model prediction
    
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns = ['price'])
    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [carat, cut, color, clarity, 
                                     depth, table, x, y, z]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    # Using predict() with new data provided by the user
    #new_prediction = clf.predict(user_encoded_df)

    # # Show the predicted species on the app
    # st.subheader("Price")

    y_pred, y_pis = clf.predict(user_encoded_df, alpha=alpha)

    # # Display results
    lower, upper = y_pis[0]
    lower_val = float(np.ravel(lower)[0])
    upper_val = float(np.ravel(upper)[0])
    # st.subheader("Price Results")
    # st.write(f"**Expected Price:** {y_pred[0]:.2f}")
    # st.write(f"**90% Prediction Interval:** ({lower_val:.2f}, {upper_val:.2f})")
    # Green header text
    st.markdown("<h2 style='color: green;'>Predicting Prices...</h2>", unsafe_allow_html=True)

    # Predicted price (big bold font)
  
    st.markdown(f"<h3>Predicted Price</h3>", unsafe_allow_html=True)
    pred_val = float(np.ravel(y_pred)[0])
    st.markdown(f"<h1 style='font-weight: bold;'>${pred_val:,.2f}</h1>", unsafe_allow_html=True)

    # Blue box for prediction interval
    st.markdown(
        f"""
        <div style="
            background-color:#e6f0ff;
            padding:10px;
            border-radius:5px;
            border:1px solid #99c2ff;
            width:fit-content;
        ">
            <b>Prediction Interval ({alpha_percent:,.2f}%):</b> [{lower_val:,.2f}, {upper_val:,.2f}]
        </div>
        """,
        unsafe_allow_html=True
    )

else:
   # Loading data
   st.success("""
    ### CSV file uploaded successfully
    """)
   user_df = pd.read_csv(diamond_file) # User provided data
   original_df = pd.read_csv('diamonds.csv') # Original data to create ML model
   
   # Dropping null values
   user_df = user_df.dropna().reset_index(drop = True) 
   original_df = original_df.dropna().reset_index(drop = True)
   
   # Remove output (species) and year columns from original data
   original_df = original_df.drop(columns = ['price'])
   # Remove year column from user data
   #user_df = user_df.drop(columns = ['year'])
   
   # Ensure the order of columns in user data is in the same order as that of original data
   user_df = user_df[original_df.columns]

   # Concatenate two dataframes together along rows (axis = 0)
   combined_df = pd.concat([original_df, user_df], axis = 0)

   # Number of rows in original dataframe
   original_rows = original_df.shape[0]

   # Create dummies for the combined dataframe
   combined_df_encoded = pd.get_dummies(combined_df)

   # Split data into original and user dataframes using row index
   original_df_encoded = combined_df_encoded[:original_rows]
   user_df_encoded = combined_df_encoded[original_rows:]

   # Predictions for user data
   #user_pred = clf.predict(user_df_encoded)
   y_pred, y_pis = clf.predict(user_df_encoded, alpha=alpha)
   lowers = y_pis[:, 0]
   uppers = y_pis[:, 1]

   # Convert to floats
   lowers = lowers.astype(float)
   uppers = uppers.astype(float)
   y_pred = y_pred.astype(float)

   # Add results to user DataFrame
   user_df["Predicted Price"] = y_pred
   user_df["Lower Price Limit"] = lowers
   user_df["Upper Price Limit"] = uppers
   
   # Display nicely
   st.subheader("📈 Predicted Diamond Prices with Intervals")
   st.dataframe(user_df, use_container_width=True)

# Showing additional items in tabs
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted vs. Actual", "Coverage Plot"])


# Tab 1: Feature Importance Visualization
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Features used in this prediction are ranked by relative importance.")

# Tab 2: Visualizing Histogram of Residuals
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residuals_dist.svg')
    st.caption("Distribution of residuals to evaluate prediction quality")

# Tab 3: Predicted vs. Actual
with tab3:
    st.write("### Plot of Predicted vs. Actual")
    st.image('pred_vs_act.svg')
    st.caption("Visual comparison of predicted actual values.")

# Tab 4: Coverage plot
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage_plot.svg')
    st.caption("Range of predictions with confience intervals.")