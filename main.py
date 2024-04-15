from fastapi import FastAPI, Request, staticfiles, responses
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from utils.constants import GroupName
from utils.utils import generate_response

app = FastAPI()

app.mount("/images", staticfiles.StaticFiles(directory="images"), name="images")
app.mount("/assets", staticfiles.StaticFiles(directory="assets"), name="static")

templates = Jinja2Templates(directory="dist")

# only for demo purposes, do not use in real production!!!
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/favicon.ico")
async def get_favicon():
    return responses.FileResponse("./dist/favicon.ico")


@app.get("/data/{group}")
async def get_test_data(group: GroupName, classes: str | None = None):
    return generate_response(group, classes)


@app.get("/{rest_of_path:path}")
async def react_app(req: Request, rest_of_path: str):
    return templates.TemplateResponse("index.html", {"request": req})
