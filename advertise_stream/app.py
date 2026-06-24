import streamlit as st
import pandas as pd
import plotly.express as px

df=pd.read_csv('Advertising_F.csv')
st.title("Advertising Effectiveness Marlin")
st.sidebar.header("Filters")
x_var=st.sidebar.selectbox("X axis",["TV","radio","newspaper"])
y_var=st.sidebar.selectbox("Y Axis",["sales"])

#show scatterplot
fig=px.scatter(df,x="TV",y='sales')
st.plotly_chart(fig)
st.plotly_chart(fig)
#fig.show()                #delete this
#Dataframe
st.write("### This is the data")
st.dataframe(df)

#Show Map
st.title("I'm the map!!!")
fig=px.scatter_mapbox(df,lat='latitude',lon='longitude',size='newspaper',
                      hover_name='City',color='sales',
                      color_continuous_scale=px.colors.sequential.Viridis)
fig.update_layout(mapbox_style='carto-positron')
st.plotly_chart(fig)
