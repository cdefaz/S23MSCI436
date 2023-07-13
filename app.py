import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import calendar
import altair as alt
import random

#Create page title
st.set_page_config(page_title="House Price Prediction", page_icon=":houses:", layout="wide")

#Select features from dataframe
raw_data_train = pd.read_csv('https://raw.githubusercontent.com/jmpark0808/pl_mnist_example/main/train_hp_msci436.csv')
df_features = raw_data_train[["Neighborhood","BedroomAbvGr", "LotArea", "YearBuilt", "YearRemodAdd", "OverallQual", "OverallCond", "FullBath", "HalfBath", "CentralAir", "TotalBsmtSF", "GarageCars", "PavedDrive", "PoolArea", "SalePrice"]]
df_graph1 = raw_data_train[["Neighborhood","SalePrice"]]

#One-hot encoding
df_one_hot = pd.get_dummies(df_features, columns = ['Neighborhood', 'CentralAir', 'PavedDrive'],dtype=np.int64)
pd.set_option('display.max_columns', None)
df = df_one_hot.select_dtypes(include = ['float64', 'int64']).fillna(0)
train, test = train_test_split(df, test_size=0.2, random_state=1) # train 1168 is rows, test is 292 rows

#Split training and testing data
columns_to_exclude = 11  # Index of the column SalePrice
X_train = train.values[:, np.r_[0:columns_to_exclude, columns_to_exclude+1:len(df.columns)]]
y_train = train.values[:, columns_to_exclude]

X_test = test.values[:, np.r_[0:columns_to_exclude, columns_to_exclude+1:len(df.columns)]]
y_test = test.values[:, columns_to_exclude]

#fit into the model & predict
reg = LinearRegression().fit(X_train, y_train)
prediction = reg.predict(X_test)

#create a pickle file
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

model=reg

#Create app header
html_price = """
<div style="background-color:#F1A378 ;padding:10px">
  <h2 style="color:white;text-align:center;">🏠 House Price Prediction ML App </h2>
  <h3 style="color:brown;text-align:center;font-size:17px;">Hello real estate brokers! Enter house information below for a fast price estimate for your client's home:</h3>
</div>
"""
st.markdown(html_price, unsafe_allow_html=True)


#Predict price based on user's 14 inputs
def predict_houseprice(Neighborhood,BedroomAbvGr,LotArea,YearBuilt,YearRemodAdd,OverallQual,OverallCond,FullBath,HalfBath,CentralAir,TotalBsmtSF,GarageCars,PavedDrive,PoolArea):
 input = np.array([
        BedroomAbvGr,
        LotArea,
        YearBuilt,
        YearRemodAdd,
        OverallQual,
        OverallCond,
        FullBath,
        HalfBath,
        TotalBsmtSF,
        GarageCars,
        PoolArea,
        0,  # Placeholder for Neighborhood_Blmngtn
        0,  # Placeholder for Neighborhood_Blueste
        0,  # Placeholder for Neighborhood_BrDale
        0,  # Placeholder for Neighborhood_BrkSide
        0,  # Placeholder for Neighborhood_ClearCr
        0,  # Placeholder for Neighborhood_CollgCr
        0,  # Placeholder for Neighborhood_Crawfor
        0,  # Placeholder for Neighborhood_Edwards
        0,  # Placeholder for Neighborhood_Gilbert
        0,  # Placeholder for Neighborhood_IDOTRR
        0,  # Placeholder for Neighborhood_MeadowV
        0,  # Placeholder for Neighborhood_Mitchel
        0,  # Placeholder for Neighborhood_NAmes
        0,  # Placeholder for Neighborhood_NPkVill
        0,  # Placeholder for Neighborhood_NWAmes
        0,  # Placeholder for Neighborhood_NoRidge
        0,  # Placeholder for Neighborhood_NridgHt
        0,  # Placeholder for Neighborhood_OldTown
        0,  # Placeholder for Neighborhood_SWISU
        0,  # Placeholder for Neighborhood_Sawyer
        0,  # Placeholder for Neighborhood_SawyerW
        0,  # Placeholder for Neighborhood_Somerst
        0,  # Placeholder for Neighborhood_StoneBr
        0,  # Placeholder for Neighborhood_Timber
        0,  # Placeholder for Neighborhood_Veenker
        0,  # Placeholder for CentralAir_N
        0,  # Placeholder for CentralAir_Y
        0,  # Placeholder for PavedDrive_N
        0,  # Placeholder for PavedDrive_P
        0   # Placeholder for PavedDrive_Y
    ]).astype(np.float64)

  # Map the corresponding neighborhood value to the appropriate placeholder
 neighborhood_mapping = {
        "Blmngtn": 11,
        "Blueste": 12,
        "BrDale": 13,
        "BrkSide": 14,
        "ClearCr": 15,
        "CollgCr": 16,
        "Crawfor": 17,
        "Edwards": 18,
        "Gilbert": 19,
        "IDOTRR": 20,
        "MeadowV": 21,
        "Mitchel": 22,
        "NAmes": 23,
        "NPkVill": 24,
        "NWAmes": 25,
        "NoRidge": 26,
        "NridgHt": 27,
        "OldTown": 28,
        "SWISU": 29,
        "Sawyer": 30,
        "SawyerW": 31,
        "Somerst": 32,
        "StoneBr": 33,
        "Timber": 34,
        "Veenker": 35
    }

 if Neighborhood in neighborhood_mapping:
        input[neighborhood_mapping[Neighborhood]] = 1

# Map the corresponding central air value to the appropriate placeholder
 air_mapping = {
        "No": 36,
        "Yes": 37
    }

 if CentralAir in air_mapping:
        input[air_mapping[CentralAir]] = 1

# Map the corresponding driveway value to the appropriate placeholder
 paved_mapping = {
        "No": 38,
        "Partially": 39,
        "Yes": 40
    }

 if PavedDrive in paved_mapping:
        input[paved_mapping[PavedDrive]] = 1

 prediction = model.predict(input.reshape(1, -1))
 pred = '{0:.{1}f}'.format(prediction[0], 2)
 return float(pred)

#Gather user inputs
def main():
  neighborhoods = (
    "CollgCr",
    "Veenker",
    "Crawfor",
    "NoRidge",
    "Mitchel",
    "Somerst",
    "NWAmes",
    "OldTown",
    "BrkSide",
    "Sawyer",
    "NridgHt",
    "NAmes",
    "SawyerW",
    "IDOTRR",
    "MeadowV",
    "Edwards",
    "Timber",
    "Gilbert",
    "StoneBr",
    "ClearCr",
    "NPkVill",
    "Blmngtn",
    "BrDale",
    "SWISU",
    "Blueste"
  )

  Neighborhood = st.selectbox("Neighborhood:",neighborhoods)
  if st.button('Client unsure of neighborhood'):
    Neighborhood = random.choice(neighborhoods)
    st.write('The random neighborhood selected is ', Neighborhood)

  BedroomAbvGr = st.text_input("Number of bedrooms above ground:", "e.x. 0, 1, 2")
  if st.button('Client unsure of bedrooms above ground'):
    BedroomAbvGrMean = str(round(df_features["BedroomAbvGr"].mean()))
    st.write('The average number of bedrooms is ', BedroomAbvGrMean)

  LotArea = st.text_input("LotArea in square feet","e.x. 1300, 8450, 14115")
  if st.button('Client unsure of lot area'):
    LotAreaMean = round(df_features["LotArea"].mean())
    st.write('The average lot area ', LotAreaMean)

  YearBuilt = st.text_input("The year the house was built:","e.x. 1880, 1997, 2001")
  if st.button('Client unsure of year house is built'):
    YearBuiltMean = round(df_features["YearBuilt"].mean())
    st.write('The average year built is ', YearBuiltMean)

  YearRemodAdd = st.text_input("The year the house was remodelled (if never remodelled, please input the build year):","e.x. 1956, 2007, 2010")
  if st.button('Client unsure of year house was remodelled'):
    YearRemodAddMean = round(df_features["YearRemodAdd"].mean())
    st.write('The average year for a remodel is ', YearRemodAddMean)

  OverallQual = st.slider('Overall quality of the house:', 1, 10)
  if st.button('Client unsure of overall quality'):
    OverallQualMean = round(df_features["OverallQual"].mean())
    st.write('The average overall quality is ', OverallQualMean)

  OverallCond = st.slider('Overall condition of the house:', 1, 9)
  if st.button('Client unsure of overall condition'):
    OverallCondMean = round(df_features["OverallCond"].mean())
    st.write('The average overall condition is ', OverallCondMean)

  FullBath = st.text_input("Number of full bathrooms:","e.x. 0, 1, 2")
  if st.button('Client unsure of number of full bathrooms'):
    FullBathMean = round(df_features["FullBath"].mean())
    st.write('The average number of full bathrooms is ', FullBathMean)

  HalfBath = st.text_input("Number of half bathrooms", "e.x. 0, 1, 2")
  if st.button('Client unsure of number of half bathrooms'):
    HalfBathMean = round(df_features["HalfBath"].mean())
    st.write('The average number of half bathrooms is ', HalfBathMean)

  CentralAir = st.radio('Does the house have central airconditioning',['Yes', 'No'])
  if st.button('Client unsure if there is central airconditioning'):
    #CentralAir = 'Yes'
    st.write('Most houses have central air conditioning')

  TotalBsmtSF = st.text_input("Total basement Squarefootage","e.x. 0, 656, 3200")
  if st.button('Client unsure of basement sq feet'):
    TotalBsmtSFMean = round(df_features["TotalBsmtSF"].mean())
    st.write('The average basement square footage is ', TotalBsmtSFMean)


  GarageCars = st.text_input('How many cars can the garage fit:', "e.x. 0, 1, 2")
  if st.button('Client unsure of how many cars the garage can fit'):
    GarageCarsMean = round(df_features["GarageCars"].mean())
    st.write('The average number of cars a garage can fit is ', GarageCarsMean)

  PavedDrive = st.radio('Is the driveway paved:',['No', 'Partially', 'Yes'])
  if st.button('Client unsure if the driveway is paved'):
    #PavedDrive = 'Yes'
    st.write('Most houses have a paved driveway')

  PoolArea = st.text_input("Pool area in squarefeet (enter 0 if there is no pool):","e.x. 0, 512, 738")
  if st.button('Client unsure of pool area'):
    df_pool_area_not_0 = df_features.loc[df_features["PoolArea"] > 0, "PoolArea"]
    PoolAreaMean = round(df_features["PoolArea"].mean())
    st.write('The average pool area (for houses with a pool) is ', PoolAreaMean)

  #Once hit the Predict button, generate a report of price prediction along with visualizations for insights
  if st.button("Predict"):
    output=predict_houseprice(Neighborhood,BedroomAbvGr,LotArea,YearBuilt,YearRemodAdd,OverallQual,OverallCond,FullBath,HalfBath,CentralAir,TotalBsmtSF,GarageCars,PavedDrive,PoolArea)

    #Report header
    price_html = f"""
    <div style="background-color:#F08080;padding:10px">
      <h2 style="color:black;text-align:center;">The predicted sale price of your house is {output}</h2>
    </div>
    """
    #st.success('The price of your inputed house is ${}'.format(output))
    st.markdown(price_html, unsafe_allow_html=True)

    #Visulaizations as insights
    st.write(
        """
    #### Should your client remodel their house before selling it?
    """
    )

    data = df.groupby(["YearRemodAdd"])["SalePrice"].mean().sort_values(ascending=True)
    df_chart = pd.DataFrame({
            'YearRemodAdd': data.index,
            'SalePrice': data.values
        })

    chart = alt.Chart(df_chart).mark_line().encode(
            x='YearRemodAdd',
            y='SalePrice',
        ).properties(
            width=600,
            height=400
        )

    st.altair_chart(chart, use_container_width=True)
    st.write(
        """
    #### Mean sale price based on neighborhoods
    """
    )

    data = df_graph1.groupby(["Neighborhood"])["SalePrice"].mean().sort_values(ascending=True)
    df_chart = pd.DataFrame({
            'Neighborhood': data.index,
            'Mean Sale Price': data.values
        })

    chart = alt.Chart(df_chart).mark_bar().encode(
            x='Neighborhood',
            y='Mean Sale Price',
        ).properties(
            width=600,
            height=400
        )

    st.altair_chart(chart, use_container_width=True)

    st.write(
            """
            ### Prices for Houses with More Bedrooms
            """
        )

    data_bedrooms = raw_data_train.groupby(["BedroomAbvGr"])["SalePrice"].mean().sort_values(ascending=True)


    chart = alt.Chart(data_bedrooms.reset_index()).mark_line().encode(
            x=alt.X('BedroomAbvGr', axis=alt.Axis(title='Bedrooms')),
            y=alt.Y('SalePrice',axis=alt.Axis(title='Mean Sale Price'))
        )

    st.altair_chart(chart, use_container_width=True)

    # Calculate the average price difference for each additional bathroom
    avg_price_diff = data_bedrooms.diff().mean()

    st.write(
    f"To increase the house price, you could consider adding an additional bathroom. On average, each additional bathroom is associated with an increase in price of approximately ${avg_price_diff:.2f}."
)

    st.write(
            """
            ### Mean Prices of Houses Sold by Month
            """
        )
    data_month = raw_data_train.groupby("MoSold")["SalePrice"].mean().sort_values(ascending=True)
    data_month.index = data_month.index.map(lambda x: calendar.month_name[x])

        # Find the most profitable month
    most_profitable_month = data_month.idxmax()

    chart = alt.Chart(data_month.reset_index()).mark_line().encode(
            x=alt.X('MoSold', axis=alt.Axis(title='Month')),
            y=alt.Y('SalePrice', axis=alt.Axis(title='Mean Sale Price'))
        ).properties(
            width=600,
            height=400
        )

    st.altair_chart(chart, use_container_width=True)

if __name__=='__main__':
  main()
