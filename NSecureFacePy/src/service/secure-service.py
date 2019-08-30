from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "Welcome to use Secure Face Service"

@app.route('/register-application', methods=['POST'])
def register_application():
	print(request.get_json())

	return ""

app.run()