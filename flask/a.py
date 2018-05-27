# --*-- encoding:utf-8 --*--
from flask import Flask

app = Flask(__name__)


# @app.route('/HelloWorld')
# def hello_world():
#     return "Hello World!"


@app.route('/user/<username>')
def show_user_profile(username):
    return 'User %s' % username


if __name__ == "__main__":
    app.run(debug=True)
