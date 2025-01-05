
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Your name is LitBot and you are a helpful literary assistant and will provide the user with help in literature."},
                {"role": "user", "content": user_message}
            ]
        )
        
        bot_response = response.choices[0].message['content']
        return jsonify({"response": bot_response})
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request."}), 500