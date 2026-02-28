from flask import Flask, request, jsonify, render_template
from model import llama_response, granite_response, mistral_response
import time

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate',methods=['POST'])
def generate():
    data=request.json
    user_message = data.get('message')
    model = data.get('model')

    if not user_message or not model:
        return jsonify({"error" : "Missing message or model selection"}), 400

    system_prompt = "You are an AI assistant helping with customer inquiries. Provide a helpful and concise response"
    
    start_time = time.time()

    try:
        if model == 'llama':
            result = llama_response(system_prompt, user_message)
        elif model == 'granite':
            result = granite_response(system_prompt, user_message)
        elif model == 'mistral':
            result = mistral_response(system_prompt, user_message)
        else:
            return jsonify({"error" : "Invalid model selection"}), 400
        result['duration'] = time.time() - start_time
        return(jsonify(result))
    except Exception as e:
        return jsonify({"error" : str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# Future Enhancements
# Implement Caching: Add a caching mechanism to improve performance for repeated queries.
# 
# Explore Advanced LangChain Features: Look into features like memory for maintaining conversation context.
# 
# Add More Models: Try integrating other models available through watsonx.ai.
# 
# Implement A/B Testing: Create a system to compare responses from different models for the same query.
# 
# Enhance Error Handling: Implement more robust error handling and logging.
# 
# Explore IBM Cloud Services: Consider integrating other IBM Cloud services to expand your application’s capabilities.