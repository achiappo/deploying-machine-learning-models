from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the House Prices API!"}

@app.get("/square")
async def square(value: int):
    result = value ** 2
    return {"square": result}