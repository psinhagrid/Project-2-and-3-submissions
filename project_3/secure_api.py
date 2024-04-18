import os 
from fastapi import FastAPI, HTTPException
import testing_main
import sentence_checker
import jwt
import datetime

# uvicorn secure_api:app --reload
# /Users/psinha/Desktop/Structred Folder/Project_3(API_and_Inferences)
# http://127.0.0.1:8000/docs#/default/read_items_items__get

app = FastAPI()

# Define a secret key for JWT token encoding and decoding
SECRET_KEY = 'your_secret_key_here'

# Define a function to generate a JWT token with the desired payload
def generate_jwt_token(user_id: int, username: str) -> str:
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # Expiration time
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token

# Generate JWT token during startup
jwt_token = generate_jwt_token(user_id=123, username='example_user')
print("JWT Token:", jwt_token)



@app.get("/items/")
def read_items(Question: str = "", Image_Link: str = "", jwt_token: str = ""):
    # Check if JWT token is provided
    if not jwt_token:
        raise HTTPException(status_code=401, detail="JWT token is required")

    # Decode and verify JWT token
    try:
        decoded_payload = jwt.decode(jwt_token, SECRET_KEY, algorithms=['HS256'])
        user_id = decoded_payload.get('user_id')
        username = decoded_payload.get('username')
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="JWT token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid JWT token")

    # Check sentence validity
    if sentence_checker.sentence_checking(Question) == "invalid sentence":
        raise HTTPException(status_code=400, detail="Invalid sentence")

    # Check URL security
    if sentence_checker.URL_check(Image_Link) == "insecure link":
        raise HTTPException(status_code=400, detail="Insecure link")

    # Check URL format
    if sentence_checker.URL_check(Image_Link) == "invalid link format":
        raise HTTPException(status_code=400, detail="Invalid link format")

    # Proceed with the request
    result = testing_main.class_predictor(Question, Image_Link, choice=3)
    return {"result": result}