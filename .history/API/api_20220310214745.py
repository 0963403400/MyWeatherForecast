from crypt import methods
from flask import Flask

app=Flask(__name__)

@app.route('/',methods=['GET'])
def api():
    return{
        "Tuan":"DepZai",
        "Tittle":"VipPro",
    }