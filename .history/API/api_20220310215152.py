from crypt import methods
from distutils.log import debug
from flask import Flask


app=Flask(__name__)

@app.route('/',methods=['GET'])
def api():
    return{
        "Tuan":"DepZai",
        "Tittle":"VipPro",
    }