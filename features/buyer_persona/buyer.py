import streamlit as st
import os
import openai

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Your question"):
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.messages.append({"role":"user","content":prompt})

# with st.chat_message("ai"):
#     message_placeholder = st.empty()
#     full_response = ""
#     for response in openai.ChatCompletion.create(
#         model = st.session_state["openai_model"],
#         messages = [
#             {"role": m["role"],"content":m["content"]}
#             for m in st.session_state.messages
#         ],
#         stream = True
#     ):
#         full_response += response.choices[0].delta.get("content","")
#         message_placeholder.markdown(full_response + "|")
#     message_placeholder.markdown(full_response)
# st.session_state.messages.append({"role":"ai","content":full_response})


    

def buyer_persona():

    openai.api_key = st.secrets["OPENAI_API_KEY"]
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    st.title('Buyer Persona' + ":person_with_blond_hair:")
    product = st.text_input("What product do you sell?")
    place = st.text_input("Where do you sell your product?")

    prompt_template = f"""Please ignore all previous instructions. Please respond only in the english language. You are a marketing researcher that writes fluent english. You have a Conversational tone of voice. You have a Creative writing style. Your task is to generate a detailed USER PERSONA for a business that sells {product} in {place}. First write "User Persona creation for {product} in {place}" as the heading. Now create a subheading called "Demographics". Below, you need to create a table with the 2 columns and 7 rows with the following format: Column 1 = Data points (Name, Age, Occupation, Annual income, Marital status, Family situation, Location), Column 2 = Answers for each data point in Column 1 based on the specific market "{product}". Now create a subheading called "USER DESCRIPTION". Below this generate a summary of the user persona in no more than 500 characters. Now create a subheading called "PSYCHOGRAPHICS". Below this you need to create a table with 2 columns and 9 rows with the following format: Column 1 = Data points (Personal characteristics, Hobbies, Interests, Personal aspirations, Professional goals, Pains, Main challenges, Needs, Dreams), Column 2 = Answers for each data point in Column 1 based on the specific market "{product}". Now create a subheading called "SHOPPING BEHAVIORS". Below this you need to create a table with 2 columns and 8 rows with the following format: Column 1 = Data points (Budget, Shopping frequency, Preferred channels, Online behavior, Search terms, Preferred brands, Triggers, Barriers), Column 2 = Answers for each data point in Column 1 based on the specific market "{product}". Please make sure that your response is structured in 4 separate tables and has a separate row for each data point. Do not provide bullet points. Do not self reference. Do not explain what you are doing."""
    

    pressed = st.button("Process")
    if product and place and pressed:
        prompt = prompt_template
        with st.chat_message("ai"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model = st.session_state["openai_model"],
                messages = [{"role":"user","content":prompt}],
                stream = True
            ):
                full_response += response.choices[0].delta.get("content","")
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)


