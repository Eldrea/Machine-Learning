from flask import request, abort, make_response
from flask import Flask
from flask import jsonify
import math
# from sklearn.externals import joblib
import joblib
from joblib import load
from json import dumps


def create_app():
    app = Flask(__name__)

    def load_global_data():
        global model
        model = load('chekin_model.mdl')
    load_global_data()
    return app


app = create_app()


@app.route('/')
def hello_world():
    return 'I am working! (/api/divmod?a=..&b=..)'

# # str(divmod(a, b))
# @app.route('/api/divmod', methods=['GET'])
# def predict():
#     a = request.args.get('a', type=int)
#     b = request.args.get('b', type=int)
#     if a is None:
#         bad_request('a not specified')
#     if a < 0:
#         bad_request('a should be greater then 0')
#     # get prediction
#     return '{0} , {1}'.format(a,b)

@app.route('/api/predict', methods=['GET'])
def predict():
    r = request.args.get('r', type=int)
    g = request.args.get('g', type=int)
    b = request.args.get('b', type=int)

    X=[[r,g,b]]
    y_pred = model.predict(X)
    if ((r is None) or (g is None) or (b is None)):
        bad_request('some parameters not specified')
    if ((r< 0) or (g< 0) or (b < 0)):
        bad_request('parameters should be greater then 0')
    return str(y_pred)

def bad_request(message, code=400):
    abort(make_response(jsonify(message=message), code))

#/api/predict?r=141&g=250&b=0
app.run()