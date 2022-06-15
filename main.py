from typing import Union
from fastapi import FastAPI, File, Form, UploadFile
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hola" : "World"}

@app.get("/users/{user_id}")
def userFunc(user_id):
    return f"El id del usuario es: {user_id}"

@app.post("/upload_photo")
def upload_photo(
    file: UploadFile = File(...)
    #first: str = Form(...),
    #second: str = Form("default value")
):
    print("Ingresando...")
    return {
        "name": file.filename,
        "type": file.content_type,
        #"first": first,
        #"second": second
    }
