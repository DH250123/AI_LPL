from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "こんにちは、Azure App Serviceからのメッセージです！"

if __name__ == '__main__':
    app.run()
