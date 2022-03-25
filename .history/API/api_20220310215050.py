from crypt import methods
from distutils.log import debug
from flask import Flask
from Tuan_Forescast import forecast

app=Flask(__name__)

@app.route('/',methods=['GET'])
def api():
    return{
        "Tuan":"DepZai",
        "Tittle":"VipPro",
    }