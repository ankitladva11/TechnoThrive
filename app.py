import streamlit as st
import pandas as pd
import sys
from features.pdf_analysis.pdf_analysis import pdfAnalysis
from features.upload_pdf.upload import upload_pdfs
from features.upload_csv.csv import upload_csv



with st.sidebar:
    st.image('logo.jpg')
    page = st.radio('NAVIGATION',['Upload PDF','Question PDFs','Question CSVs'])
    st.info('Auto ML is an application built with the objective of automating the process of ML pipelining')

if page:

    if page=='Upload PDF':
        pdfAnalysis()
    if page=='Question PDFs':
        upload_pdfs()
    if page=='Question CSVs':
        upload_csv()

        


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


