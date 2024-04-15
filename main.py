from fastapi import FastAPI, staticfiles
from fastapi.middleware.cors import CORSMiddleware

from utils.constants import GroupName
from utils.utils import generate_response

app = FastAPI()

app.mount("/images", staticfiles.StaticFiles(directory="images"), name="images")


# only for demo purposes, do not use in real production!!!
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/data/{group}")
async def get_test_data(group: GroupName, classes: str | None = None):
    return generate_response(group, classes)
