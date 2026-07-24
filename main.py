from fastapi import FastAPI
from routers.player import router
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {'message': 'Hello World'}

app.include_router(router, prefix="/players")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)