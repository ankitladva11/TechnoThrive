# Libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import datetime
# from utils.model import *
# from utils.kmeans_feature_importance import KMeansInterp
from features.customer_segmentation_true.utils.model import *
from features.customer_segmentation_true.utils.kmeans_feature_importance import KMeansInterp
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import StandardScaler    
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(
    page_title="Customer Behavior Analysis",
    page_icon="📊",
    layout="wide"
)

 # Cache the function
st.title("Customer Behavior Analysis", anchor="customer-behavior-analysis")
# Upload file
uploaded_file = st.file_uploader("Please Import Your Transaction Data", type="csv")
if uploaded_file is None:
    data = pd.read_csv("D:/Programming/Streamlit/TechnoThrive/features/customer-segmentation-streamlit-master/data/data.csv", encoding='latin-1')
    df1 = preprocessing(data)
    st.subheader("Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Transactions", df1.InvoiceNo.nunique())
    col2.metric("Number of Customers", df1.CustomerID.nunique())
    col3.metric("Average Revenue", f'${round(df1.Total.sum()/df1.InvoiceNo.nunique(),2)}')
    c = st.empty()
    # Add image
    url = "https://www.kaggle.com/carrie1/ecommerce-data"
    st.image("D:/Programming/Streamlit/TechnoThrive/features/customer-segmentation-streamlit-master/data/image.png", width=750)
    st.caption("🗳 Source : Kaggle [link](%s) " % url, unsafe_allow_html=True)

else:
    df1 = pd.read_csv(uploaded_file, encoding='latin-1')
    c = st.empty()
    c.dataframe(df1.head(7))
    #c = st.empty()
    #st.write("Here is a sample of your Transaction Dataset")
    #c.dataframe(df.head(50))
    #st.write("Here is your transaction dataset")
    #st.table(df.head(7))

#df = df1.copy()

def home():
    st.title("Customer Behavior Analysis", anchor="rfm-segmentation-report")
    st.markdown("In this project, we will analyze customer behavior using :")
    st.markdown(" * Exploratory Data Analysis via Pandas Profiling")
    st.markdown(" * Customer Segmentation using RFM Analysis")
    st.markdown(" * Market Basket Analysis")
    st.markdown(" * Customer Lifetime Value Prediction using Buy Till You Die Model")
    
# def RFM():
#     st.title("RFM Segmentation Report", anchor="rfm-segmentation-report")
#     st.markdown("RFM Segmentation is method of customer segmentation that uses three key factors: recency, frequency, and monetary value. Recency is the number of days since the last purchase. Frequency is the number of purchases in a given time period. Monetary value is the total amount of money spent in a given time period.")
#     st.markdown(" * R is the number of days since the last purchase.")
#     st.markdown(" * F is the number of purchases in a given time period.")
#     st.markdown(" * M is the total amount of money spent in a given time period.")
#     df = df1.copy()
#     st.subheader("Filtered Transaction Dataset")
#     c1 = st.empty()
#     c1.dataframe(df.head(50))

# # Create sidebar to select features
#     st.sidebar.title("Select Features to Filter Data")
#     st.sidebar.subheader("📅 Date Variable")
#     date_col = st.sidebar.selectbox("Select Date", df.columns)

#     # Create date range selection
#     st.sidebar.subheader("Date Range")
#     min_date, max_date = range_selection(df, 'myd')
#     date = st.sidebar.date_input("Select your date range", (min_date, max_date))
#     if date[0] >= min_date and date[1] <= max_date:
#         df = df[(df['InvoiceDate'] >= pd.to_datetime(date[0])) & (df['InvoiceDate'] <= pd.to_datetime(date[1]))]
#     else:
#         st.error(f'Error, your chosen date is out of data range, our dataset record the transaction between {df.InvoiceDate.min()} to {df.InvoiceDate.max()}', icon="🚨")

#     st.sidebar.subheader("🛒 Transaction ID")
#     transaction_id = st.sidebar.selectbox("Select Transaction ID:", df.columns)

#     st.sidebar.subheader("👥 Customer ID")
#     customer_id = st.sidebar.selectbox("Select Customer ID:", df.columns)

#     st.sidebar.subheader("💰 Revenue Variable")
#     revenue = st.sidebar.selectbox("Select Revenue Variable", df.columns)

#     st.sidebar.subheader("🌍 Region Variable")
#     region = st.sidebar.selectbox("Select Region Variable", df.columns)

#     st.sidebar.subheader("🏠 Country Filter")
#     country = st.sidebar.selectbox("Select specific counttry you want to analyze or select none for all country:", np.append(df[region].unique(), "None"))
#     if country == "None":
#         pass
#     else:
#         df = df[df[region] == country]

#     # Add state selection based on country value with none as default
#     c1.dataframe(df.sort_values(revenue, ascending = False))
#     st.caption("Download Your Transaction Dataset")
#     st.download_button("⬇️ Download", df.to_csv(index=False), "transaction_result.csv")

#     st.subheader("📊 Summary Metrics")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Number of Transactions", df[transaction_id].nunique())
#     col2.metric("Number of Customers", df[customer_id].nunique())
#     col3.metric("Average Revenue", f'${round(df[revenue].sum()/df[transaction_id].nunique(),2)}')

#     cola, colb, colc = st.columns(3)
#     cola.metric("Highest Revenue", f'${round(df[revenue].max(),2)}')
#     colb.metric("Lowest Revenue", f'${round(df[revenue].min(),2)}')
#     colc.metric("Most Frequent Bought Item", ', '.join(df['Description'].value_counts().index[:3].tolist()))

#     st.subheader("📈 Revenue Distribution")
#     # Create revenue distribution chart
#     fig = px.histogram(df.groupby(transaction_id, as_index=False)[revenue].sum(), x=revenue, nbins=50, title="Revenue Distribution")
#     fig.update_layout(xaxis_title="Total Revenue ($)", yaxis_title="Number of Transactions")
#     st.plotly_chart(fig, use_container_width=True)


#     # Create RFM table from model.py
#     rfmTable = create_rfm_table(df, date_col, transaction_id, customer_id)
#     st.header("📊 Recency, Frequency, and Monetary (RFM) Table", anchor="rfm-analysis")

#     # Add slider to select the range of recency, frequency, and monetary value
#     st.caption("RFM Filter")

#     # Create slider to filter RFM table with None as default
#     recency = st.slider("Select Recency Range", 0, 1000, (0, 1000))
#     st.caption("You select recency from {} to {} days".format(recency[0], recency[1]))
#     frequency = st.slider("Select Frequency Range", 0, 1000, (0, 1000))
#     st.caption("You select frequency from {} to {} days".format(frequency[0], frequency[1]))
#     monetary = st.slider("Select Monetary Range", 0, 20000, (0, 20000))
#     st.caption("You select monetary from {} USD to {} USD".format(monetary[0], monetary[1]))
    
#     # Filter RFM table that contains range of recency, frequency, and monetary value
#     rfmTable = rfmTable[(rfmTable['recency'] >= recency[0]) & (rfmTable['recency'] <= recency[1])]
#     rfmTable = rfmTable[(rfmTable['frequency'] >= frequency[0]) & (rfmTable['frequency'] <= frequency[1])]
#     rfmTable = rfmTable[(rfmTable['monetary'] >= monetary[0]) & (rfmTable['monetary'] <= monetary[1])]
#     rfmTable = rfmTable.sort_values('monetary', ascending=False)

#     r1_col1, r1_col2 = st.columns(2)
#     with r1_col1:
#         st.subheader("📊 RFM Table")
#         st.dataframe(rfmTable)
#         st.caption("Download Your RFM Table")
#         st.download_button("⬇️ Download", rfmTable.to_csv(index=False), "rfm_result.csv")
#     with r1_col2:
#         st.subheader("📊 Summary Metrics")
#         st.write("Here is the summary metrics of your filtered RFM table")
#         st.dataframe(rfmTable.describe())
    
#     st.header("📈 RFM Distribution", anchor="rfm-distribution")
#     st.markdown("Here is the distribution of your filtered RFM table")
#     r2_col1, r2_col2, r2_col3 = st.columns(3)
#     with r2_col1:
#         #st.caption("Recency Distribution")
#         fig = px.histogram(rfmTable, x='recency', nbins=50, title="Recency Distribution")
#         fig.update_layout(xaxis_title="Recency (Days)", yaxis_title="Number of Customers")
#         st.plotly_chart(fig, use_container_width=True)
#     with r2_col2:
#         #st.caption("Frequency Distribution")
#         fig = px.histogram(rfmTable, x='frequency', nbins=50, title="Frequency Distribution")
#         fig.update_layout(xaxis_title="Frequency", yaxis_title="Number of Customers")
#         st.plotly_chart(fig, use_container_width=True)
#     with r2_col3:
#         #st.caption("Monetary Distribution")
#         fig = px.histogram(rfmTable, x='monetary', nbins=50, title="Monetary Distribution")
#         fig.update_layout(xaxis_title="Monetary (USD)", yaxis_title="Number of Customers")
#         st.plotly_chart(fig, use_container_width=True)

#     st.header("📈 RFM Segmentation", anchor="rfm-segmentation")
#     # Add selecter to select the number of segment and technique
#     st.markdown("There are two techniques you can use to segment your RFM table, K-Means and Quantile.\
#         (1) K-Means technique use all RFM features to segment your customers by scaling it first.\
#             (2) Quantile technique score the RFM features and then use the score to segment your customers by using K-Means clustering.")
#     segment = st.selectbox("Select the number of segment you want to create:", [2, 3, 4, 5])
#     technique = st.selectbox("Select the technique you want to use:", ["K-Means", "Quantile"])
#     if technique == "K-Means":
#         rfm_segment, feat_imp = rfm_segmentation(rfmTable, 1, segment)
#     else:
#         rfm_segment, feat_imp = rfm_segmentation(rfmTable, 2, segment)

#     ra_c1, ra_c2 = st.columns(2)
#     with ra_c1:
#         st.subheader("📊 RFM Segmentation Table")
#         st.dataframe(rfm_segment)
#         st.caption("Download Your Segmentation Result")
#         st.download_button("⬇️ Download", rfm_segment.to_csv(index=False), "rfm_segmentation_result.csv")
#     with ra_c2:
#         summary = rfm_segment.groupby('RFM_Segment').agg({
#             'recency': 'mean',
#             'frequency': 'mean',
#             'monetary': 'mean',
#             'CustomerID': 'count'
#         }).reset_index().rename(columns={'CustomerID': 'N'})
#         st.subheader("📊 Summary Metrics")
#         st.write("Here is the summary metrics of your segmentation result")
#         st.dataframe(summary)

#     st.header("📈 RFM Segmentation Exploration", anchor="rfm-segmentation-distribution")
#     tab0, tab1, tab2, tab3, tab4 = st.tabs(["Cluster Distribution","1️⃣ Recency vs Frequency", "2️⃣ Recency vs Monetary", "3️⃣ Frequency vs Monetary", "4️⃣ 3D Plot"])
#     with tab0:
#         fig = px.bar(summary, y='RFM_Segment', x="N", orientation='h', title="Segment Distribution")
#         fig.update_layout(xaxis_title="Number of Customers", yaxis_title="Segment")
#         st.plotly_chart(fig, use_container_width=True)
#     with tab1:
#         # sort categories order from 1 to n
#         fig = px.scatter(rfm_segment, x='recency', y='frequency', color='RFM_Segment', title="Recency vs Frequency",
#                          category_orders={"RFM_Segment": [str(i) for i in range(1, segment+1)]})
#         fig.update_layout(xaxis_title="Recency (Days)", yaxis_title="Frequency")
#         # sort the labels on the legend from 1 to 5
#         st.plotly_chart(fig, use_container_width=True)
#     with tab2:
#         fig = px.scatter(rfm_segment, x='recency', y='monetary', color='RFM_Segment', title="Recency vs Monetary",
#                          category_orders={"RFM_Segment": [str(i) for i in range(1, segment+1)]})
#         fig.update_layout(xaxis_title="Recency (Days)", yaxis_title="Monetary (USD)")
#         st.plotly_chart(fig, use_container_width=True)
#     with tab3:
#         fig = px.scatter(rfm_segment, x='frequency', y='monetary', color='RFM_Segment', title="Frequency vs Monetary",
#                          category_orders={"RFM_Segment": [str(i) for i in range(1, segment+1)]})
#         fig.update_layout(xaxis_title="Frequency", yaxis_title="Monetary (USD)")
#         st.plotly_chart(fig, use_container_width=True)
#     with tab4:
#         fig = px.scatter_3d(rfm_segment, x='recency', y='frequency', z='monetary', color='RFM_Segment', title="3D Plot",
#                             category_orders={"RFM_Segment": [str(i) for i in range(1, segment+1)]})
#         #fig.update_layout(xaxis_title="Recency (Days)", yaxis_title="Frequency", zaxis_title="Monetary (USD)")
#         st.plotly_chart(fig, use_container_width=True)

#     # Print feature importance
#     st.header("📈 Feature Importance", anchor="feature-importance")
#     st.markdown("K-Means aim is to minimize the Within-Cluster Sum of Squares and consequently the Between-Cluster Sum of Squares, and assuming that the distance metric used is euclidean.\
#                 Then we will try to find the feature d_i that was responsible for the highest amount of WCSS (The sum of squares of each data point distance to its cluster centroid) minimization through finding the maximum absolute centroid dimensional movement.")
#     for cluster_label, feature_weights in feat_imp.items():
#         st.subheader(f"📊 Customer Segment {cluster_label+1}")
#         df_feat_imp = pd.DataFrame(feature_weights, columns=['feature', 'weight'])
#         fig = px.bar(df_feat_imp, x='feature', y='weight', title="Feature Importance")
#         fig.update_layout(xaxis_title="Feature", yaxis_title="Weight")
#         st.plotly_chart(fig, use_container_width=True)
    

def data_exploration():
    df = df1.copy()
    st.title("Data Exploration via Pandas Profiling", anchor="data-exploration")
    st.caption("Please wait for a few seconds for the data exploration to be completed")
    df_profile = ProfileReport(df, explorative=True)
    st_profile_report(df_profile)

# def market_basket():
#     df = df1.copy()
#     st.dataframe(df)
#     st.title("Market Basket Analysis", anchor="market-basket-analysis")
#     st.markdown("Market Basket Analysis is a data science technique to identify the buying pattern of the customers.\
#                 It aims to uncover the item groups that are frequently purchased together.")
#     st.subheader("Your Dataset", anchor="data")
#     st.sidebar.subheader("🌍 Location Filter")
#     location = st.sidebar.selectbox("Select specific location you want to analyze or select none for all location:", np.append(df['Country'].unique(), "None"))
#     if location == "None":
#         pass
#     else:
#         df = df[df['Country'] == location]
#     st.sidebar.subheader("🛒 Transaction ID")
#     transaction_id = st.sidebar.selectbox("Select Transaction ID:", df.columns)
#     st.sidebar.subheader("🧺 Item Description")
#     item = st.sidebar.selectbox("Select Item Description:", df.columns)
#     st.sidebar.subheader("👤 Customer ID")
#     customer_id = st.sidebar.selectbox("Select Customer ID:", df.columns)
#     st.sidebar.subheader("🛍 Quantity")
#     quantity = st.sidebar.selectbox("Select Quantity:", df.columns)
#     st.sidebar.subheader("💰 Revenue Variable")
#     revenue = st.sidebar.selectbox("Select Revenue Variable", df.columns)
#     c2 = st.empty()
#     c2.dataframe(df.head(50))

#     st.subheader("Item Bought Insights", anchor="item-bought-insights")
#     st.markdown("This section will show you the item bought insights through visualization.")
#     st.markdown("📊 **Most Bought Based by The Highest Number of Customers**")
#     fig = px.bar(df.groupby(item).agg({customer_id: 'nunique'}).reset_index().sort_values(by=customer_id, ascending=False)[:10],
#                 x=customer_id, y=item, title="Top 10 Items Bought by Most Customers")
#     fig.update_layout(yaxis_title="Item", xaxis_title="Number of Customers", yaxis = dict(autorange="reversed"))
#     st.plotly_chart(fig, use_container_width=True)
    
#     st.markdown("📊 **Most Bought Item Based on Quantity**")
#     fig = px.bar(df.groupby(item).agg({quantity: 'sum'}).reset_index().sort_values(by=quantity, ascending=False)[:10],
#                 x=quantity, y=item, title="Top 10 Items Bought by Most Quantity")
#     fig.update_layout(yaxis_title="Item", xaxis_title="Quantity", yaxis = dict(autorange="reversed"))
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown("📊 **Most Bought Item Based on Revenue**")
#     fig = px.bar(df.groupby(item).agg({revenue: 'sum'}).reset_index().sort_values(by=revenue, ascending=False)[:10],
#                 x=revenue, y=item, title="Top 10 Items Bought by Most Revenue")
#     fig.update_layout(yaxis_title="Item", xaxis_title="Revenue", yaxis = dict(autorange="reversed"))
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown("📊 **Top Ten First Choice**")
#     df_sort = df.sort_values(by = [transaction_id, 'InvoiceDate'])
#     first_choice = df_sort.groupby(customer_id)[item].first().value_counts().reset_index().rename(columns={'index': item, item: 'count'})
#     fig = px.bar(first_choice[:10], x='count', y=item, title="Top Ten First Choice")
#     fig.update_layout(yaxis_title="Item", xaxis_title="Count", yaxis = dict(autorange="reversed"))
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("📦 Frequently Bought Together", anchor="frequently-bought-together")
#     st.markdown("This section will show you the frequently bought together by using association rules MBA.")
#     st.markdown("Please input how percentage of support you want to use for the association rules.")
#     # Create a text box for support
#     support = st.number_input("Insert support proportion (0-1): ", min_value=0.0, max_value=1.0, value=0.01, step=0.001)
#     st.write("You selected:", support)
#     st.subheader("📊 Basket Dataset", anchor="basket-analysis")
#     basket_df = df.groupby([transaction_id, item])[item].count().unstack().reset_index().fillna(0).set_index(transaction_id)
#     basket_df = basket_df.applymap(encode_units)
#     st.dataframe(basket_df.head(50))

#     st.subheader("📊 Top 50 Association Rules", anchor="association-rules")
#     st.warning("if association rules are not shown, please change the support value")
#     rules = mba(basket_df, support)
#     st.dataframe(rules.head(50))

#     st.subheader("📊 Market Basket Recommendation", anchor="market-basket-recommendation")
#     #st.markdown("This section will show you the market basket recommendation based on the association rules.")
#     #st.markdown("Please input the item you want to recommend.")
#     #item_selected = st.selectbox("Select Item:", df[item].unique())
#     #st.write("Based on the association rules, we recommend you to buy the following items:")
#     #recommendation = frequent_bought_recommender(basket_df, item_selected, sp)
#     #st.write(recommendation)

def churn():
    df = df1.copy()
    st.title("Customer Lifetime Value", anchor="customer-lifetime-value")
    st.write("Predict CLTV with Buy Till You Die Model aka Beta Geometric/Negative Binomial Distribution Model (BG/NBD)")
    st.error("This is still under development Hehe Will be updated soon 👻")

def runner():
    side = st.sidebar.selectbox("Select your page", ["Home", "Data Exploration", "RFM Analysis", "Market Basket Analysis", "Customer Lifetimes Value"])
    if side == "Home":
        home()
    elif side == "Data Exploration":
        data_exploration()
    elif side == "RFM Analysis":
        # RFM()
        print('hehe lol')
    elif side == "Market Basket Analysis":
        # market_basket()
        print('hehe lol')
    else:
        churn()