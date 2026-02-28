from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

credentials = Credentials(
    url = "https://us-south.ml.cloud.ibm.com"
)

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}

model = ModelInference(
#    model_id="ibm/granite-3-3-8b-instruct",
#    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
#    model_id="ibm/granite-13b-instruct-v2",
    model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
    params=params,
    credentials=credentials,
    project_id="skills-network"
)

text_llama = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant who provides concise and accurate answers<|eot_id>

<|start_header_id|>user<|end_header_id|>
What is the capital of Canada?<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

text_mistral = """
<s>
[INST]You are an expert assistant who provides concise and accurate answers.[/INST]

[INST]What is the capital of Canada?[/INST]
<\s>
"""

text = text_mistral
print(model.generate(text)['results'][0]['generated_text'])
