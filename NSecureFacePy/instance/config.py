from os.path import dirname, join
from dotenv import load_dotenv


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

DATABASE = 'face-auth.db'
SECRET_KEY = b'\xe3\xf7\xaf\xf102\x9e\xec\x81t\xc7\xc5\xbb\xed\xf3C\xa9VR\x1c\x943\x91o'
