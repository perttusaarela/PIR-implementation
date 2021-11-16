from flask import Flask, request, send_file, make_response, abort
from sage.misc.persist import loads, dumps

app = Flask(__name__)

stored_data = None


@app.route("/")
def main():
    """Status indicator"""
    return "Hello, world!"


@app.route("/store", methods=["POST"])
def store():

    global stored_data

    stored_data = loads(request.form.get("data"))

    return


@app.route("/retrieve", methods=["POST"])
def retrieve():

    if stored_data is None:
        abort(400)

    query = loads(request.form.get("query"))

    r = query.dot_product(stored_data)

    return dumps(r)


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)
