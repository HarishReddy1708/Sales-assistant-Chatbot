from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import logging
from utils.suggetions import get_suggested_questions
from utils.response import ask_mistral

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
""" app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SESSION_TYPE'] = 'filesystem' """
socketio = SocketIO(app, manage_session=True)


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(message):
    try:
        if isinstance(message, dict):
            menu_context = message.get('menu_context', None)
            selected_model = message.get('selected_model', None)
            user_message = message.get('message', '')
        else:
            menu_context = None
            selected_model = None
            user_message = str(message)
        
        if not user_message:
            emit('error', 'Empty message received')
            return
            
        response = ask_mistral(user_message, menu_context=menu_context, selected_model=selected_model)
        
        
        suggested_questions = get_suggested_questions(menu_context, user_message)

        
        emit('response', {
            'message': response,
            'suggested_questions': suggested_questions
        })
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        emit('error', 'Sorry, I encountered an error. Please try again.')

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)