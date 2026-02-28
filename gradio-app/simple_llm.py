#Import the necessary packages

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM
import gradio as gr

# Specify the model and project settings
# (make sure the model you wish to use is commented out, and the other models are commented)
#model_id='mistralai/mistral-small-3-1-24b-Instruct-2503' #Mixtral 8x7B model (Unsupported??)
model_id='ibm/granite-3-3-8b-instruct' # IBM Granite 3.3 8B model

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5
}

project_id="skills-network"

watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url="https://us-south.ml.cloud.ibm.com",
    project_id=project_id,
    params=parameters
)

# CLI Testing
#query = input("Please enter your query: ")
#print(watsonx_llm.invoke(query))

def generate_response(prompt_txt):
    generated_response=watsonx_llm.invoke(prompt_txt)
    return generate_response

#Create Gradio interface
chat_application=gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label='Input',lines=2,placeholder="Type your question here..."),
    outputs=gr.Textbox(label='Output'),
    title='Watsonx.ai Chatbot',
    description='Asm any question and the chatbot will try to answer'
)

chat_application.launch(server_name="127.0.0.1", server_port=7860)