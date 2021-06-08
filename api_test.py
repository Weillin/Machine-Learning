from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/data/test", methods=["POST", "GET"])
def getDataFromJson():
    try:
        if request.method == "POST":
            name = request.json["name"]
            address = request.json["address"]
            return jsonify({"code": 200, "infos": {"name": name, "address": address}})
        else:
            return jsonify({"code": 400, "infos": "调用方法错误，请使用POST方法"})

    except BaseException:
        return jsonify({"code": 400, "infos": "参数错误，请检查"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=True)
