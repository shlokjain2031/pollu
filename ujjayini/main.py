from fastapi import FastAPI

app = FastAPI(title="Pollu API", version="0.1")

@app.get("/")
def root():
    return {"message": "Hello from Ujjayini (Pollu API is alive!)"}
