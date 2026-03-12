import streamlit as st
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

st.title("Chicago PatrolIQ")

#fetch data
df_patrol = pd.read_csv(r"Data/CrimesData.csv.gz")

#scale the data
scalar = StandardScaler()
df_patrol[['X', 'Y']] = scalar.fit_transform(df_patrol[['Longitude', 'Latitude']])
df_patrol['crime_wt'] = df_patrol['crime_severity_score']
df_patrol['crime_severity_score'] = round(df_patrol['crime_severity_score']/100, 2)

@st.cache_resource
def get_model():
    return joblib.load(r"Model/model.pkl")

with st.sidebar:
    
    st.title("Navigation")
    page = st.radio(
        "Go to",
        [
            "Home Page",
            "Data Clustering",
            "PCA",
            "Temporal Pattern Clustering"
        ],
        index=0,
        width='stretch'
    )

    st.divider()

if page == "Home Page":
    st.subheader("Chicago Crime Dashboard")
    fig = px.density_map(
            df_patrol.sample(100000),
            lat="Latitude",
            lon="Longitude",
            z='crime_wt',
            radius=10,
            opacity=0.7,
            width=1000,
            height=800,
            color_continuous_scale=px.colors.sequential.YlOrRd
        )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        title="Crime Density Heatmap"
    )
    st.plotly_chart(fig)

    st.divider()

    selYear = st.selectbox("Select Year", np.sort(df_patrol['Year'].unique()), width=100)
    df = df_patrol[df_patrol['Year'] == selYear].groupby(['Month', 'Primary Type'], as_index=False)['Primary Type'].agg(['count'])
    pivot_df = df.pivot(index='Month', columns='Primary Type', values='count').reset_index()

    month_map = {
            1: 'Jan',
            2: 'Feb',
            3: 'Mar',
            4: 'Apr',
            5: 'May',
            6: 'Jun',
            7: 'Jul',
            8: 'Aug',
            9: 'Sep',
            10: 'Oct',
            11: 'Nov',
            12: 'Dec'
        }
    pivot_df['month_name'] = pivot_df['Month'].map(month_map)

    fig = px.bar(
        pivot_df,
        x='month_name',
        y=pivot_df.columns[1:],
        barmode='stack',
        labels={'month_name':'Month',"value":'Total Crimes'},
        title="Monthly crime count"
    )

    fig.update_layout(
        width=1000,
        height=600,
        legend_title="Primary Type"
    )

    st.plotly_chart(fig)

elif page == "Data Clustering":
    st.subheader("Data Clustering")
    df_patrol = df_patrol[(df_patrol['Year'] > 2019) & (df_patrol['Year'] < 2026)]

    model = get_model()
    if model == None:
            print('Could not get the model')

    df_patrol['cluster2'] = model.fit_predict(df_patrol[['X', 'Y']])
    df = df_patrol[df_patrol['cluster2'] != -1]

    centroids = (
        df
        .groupby('cluster2')[['Latitude','Longitude']]
        .mean()
        .reset_index()
    )

    # Plot scatter plot with centroids
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.scatterplot(df, x='Longitude',y='Latitude', hue='cluster2', palette='pastel', ax=ax)
    sns.scatterplot(x=centroids['Longitude'],y=centroids['Latitude'], ax=ax)
    st.pyplot(fig)

    sample_size = st.slider("Sample size for calculationg score", 10000, 100000,len(df_patrol) ,step=10000)
    if st.button("Calculate Silhouette score"):
        df_patrol['cluster2'] = model.fit_predict(df_patrol[['X','Y']])

        df = df_patrol[df_patrol['cluster2'] != -1]
        sil_score = silhouette_score(df[['X','Y']], df['cluster2'], sample_size=sample_size)
        st.info(f"Silhouette score : {round(sil_score, 2)}")

elif page == "PCA":
    st.subheader("Principal Component Analysis")
    num_cols = ['Beat','District', 'Ward', 'Community Area', 'X Coordinate', 'Y Coordinate',
       'Latitude', 'Longitude', 'Day_of_Week', 'Hour', 'Month', 'Year', 'crime_severity_score']

    df_patrol[num_cols] = scalar.fit_transform(df_patrol[num_cols])

    #one hot encoding for season
    df_patrol = pd.concat([df_patrol, pd.get_dummies(df_patrol['Season'], drop_first=True, dtype=int)], axis=1)

    AllCols = ['Arrest', 'Domestic', 'Beat',
        'District', 'Ward', 'Community Area', 
        'X Coordinate', 'Y Coordinate',
       'Latitude', 'Longitude',
         'Day_of_Week', 'Hour', 'Month', 'Year',
       'Is_Weekend', 'crime_severity_score', 
       'Spring', 'Summer', 'Winter']
    cols = [#'Arrest', 'Domestic', 'Beat',
        'District', 'Ward', #'Community Area', 
        'X Coordinate', 'Y Coordinate',
       #'Latitude', 'Longitude',
         'Day_of_Week', 'Hour', 'Month', 'Year',
       'Is_Weekend', #'crime_severity_score', 
       'Spring', 'Summer', 'Winter']
    
    # Sidebar for Hyperparameters
    st.sidebar.header("Model Parameters")

    cols = st.multiselect("Select features: ", AllCols, default=cols)
    n_components = st.sidebar.slider("n_components", 1, 5, 4)

    # Train model
    model = PCA(n_components=n_components)
    X_pca = model.fit_transform(df_patrol[cols])
    col_Names = []
    for i in range(1,(n_components+1)):
        col_Names.append('PC' + str(i))
    loadings = pd.DataFrame(
        model.components_.T,
        columns=col_Names,
        index=cols
    )
    st.dataframe(loadings)

    st.divider()

    importance = np.sum(
        np.abs(model.components_.T) * model.explained_variance_ratio_,
        axis=1
    )

    feature_importance = pd.DataFrame({"Features":cols, "Importance": importance}).sort_values(['Importance'],ascending=False)

    st.dataframe(feature_importance, hide_index=True)

elif page == "Temporal Pattern Clustering":
    st.subheader("Temporal Pattern Clustering")
    st.write("Using Kmeans, we are trying to find the 10 Temporal Pattern during which the crime is high")
    scalarT = StandardScaler()
    cols = ['Month','Day_of_Week', 'Hour']
    arr_timeDim= scalarT.fit_transform(df_patrol[cols])
    df_timeDim = pd.DataFrame(arr_timeDim, columns=cols)

    model = KMeans(n_clusters=10)
    model.fit_predict(df_timeDim[cols])

    df_timeDimCenter = pd.DataFrame(model.cluster_centers_, columns=cols)

    arr_timeDimCenter = scalarT.inverse_transform(df_timeDimCenter)

    df_timeDimCenter = pd.DataFrame(arr_timeDimCenter, columns=cols)
    df_timeDimCenter = df_timeDimCenter.round(0)
    st.dataframe(df_timeDimCenter, hide_index=True)

    st.write("We can conclude that the crime is high during:")
    st.write("Month: March - April, Sept - Oct")
    st.write("Day of the Week: Monday, Thursday and Friday")
    st.write("Hours: 5.00-7.00 PM")

