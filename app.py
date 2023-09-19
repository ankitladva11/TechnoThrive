import streamlit as st
import pandas as pd
import sys
# from features.pdf_analysis.pdf_analysis import pdfAnalysis
from features.upload_pdf.upload import upload_pdfs
from features.upload_csv.csv import upload_csv
from features.buyer_persona.buyer import buyer_persona
from features.customer_segmentation_true.app import runner

st.set_page_config(page_title="SamahGrah",page_icon="💰")

with st.sidebar:
    st.image('logo.gif')
    page = st.radio('NAVIGATION',['AI Powered PDFs chatbot','CSVs data extractor','Buyer Persona','Consumer profiling','Your Financial Analyst'])
    st.info('SamahGrah is a one-stop solution to all the financial queries')

if page:
    if page=='AI Powered PDFs chatbot':
        upload_pdfs()
    if page=='CSVs data extractor':
        upload_csv()
    if page=='Buyer Persona':
        buyer_persona()
    if page=="Consumer profiling":
        runner()


        


    # if page=='Profiling':
    #     st.title('Exploratory Data Analysis')
    #     st.subheader('Profile Report for the uploaded dataset')
    #     # profile_report = df.profile_report()
    #     # st_profile_report(profile_report)

    # if page=='Apply ML models':
    #     st.title('Apply Machine Learning Models')
    #     target = st.selectbox('Select the target column',df.columns)
    #     # b = st.button('Train models')
    #     # if b:
    #     #     setup(df,target=target)
    #     #     setup_df = pull()
    #     #     st.info('The setup configuartion for the models:')
    #     #     st.dataframe(setup_df)
    #     #     best_model = compare_models()
    #     #     compare_df = pull()
    #     #     st.info('The ML models trained:')
    #     #     st.dataframe(compare_df)
    #     #     save_model(best_model,'trained_model')


    # if page=='Download Ml model':
    #     st.title('Download the trained model')

    #     with open('trained_model.pkl','rb') as f:
    #         st.download_button('Download model',f,'model.pkl')


