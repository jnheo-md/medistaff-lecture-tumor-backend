from fastapi import FastAPI, File, UploadFile
import os
import torch
import torchio as tio
import monai
import matplotlib.pyplot as plt
from fastapi.staticfiles import StaticFiles
import aiofiles
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULT_PATH = "/home/ec2-user/medistaff/static/results"
MODEL_PATH = "/home/ec2-user/medistaff/model.pth"
UPLOAD_PATH = "/home/ec2-user/medistaff/uploads"

def get_segmentation(model, data, device):
    model.eval()
    model.to(device)
    input = torch.cat([data[sequence]['data'].unsqueeze(0) for sequence in ['t1', 't2', 't1ce', 'flair']], dim=1).to(device) 
    output = model(input).cpu().detach()
    pred = torch.nn.functional.one_hot(output.argmax(dim=1).squeeze(0)).permute(3,0,1,2) 
    return tio.LabelMap(tensor=pred)


async def upload_file(file) :
    out_file_path = f'{UPLOAD_PATH}/{time.time()}-{file.filename}'
    async with aiofiles.open(out_file_path, 'wb') as out_file:
        while content := await file.read(1024):  # async read chunk
            await out_file.write(content)  # async write chunk
    return out_file_path



@app.post("/predict")
async def root(file_t1 : UploadFile = File(...), file_t2 : UploadFile = File(...),file_t1ce : UploadFile = File(...),file_flair : UploadFile = File(...)) :

    ## upload files 
    t1_path = await upload_file(file_t1)
    t2_path = await upload_file(file_t2)
    t1ce_path = await upload_file(file_t1ce)
    flair_path = await upload_file(file_flair)

    model = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=4, out_channels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    subject = tio.Subject(
        t1=tio.ScalarImage(t1_path),
        t2=tio.ScalarImage(t2_path),
        t1ce=tio.ScalarImage(t1ce_path),
        flair=tio.ScalarImage(flair_path),
    )
    subjects = [subject]
    transforms = [
        tio.ToCanonical(),
        tio.Resample(3),
        tio.CropOrPad((64,64,48)),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ]
    transform = tio.Compose(transforms)

    # SubjectsDataset 객체 만들기
    dataset = tio.SubjectsDataset(subjects, transform=transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg = get_segmentation(model, dataset[0], device)

    THIS_TIME = time.time()

    seg.to_gif(axis=2, duration=10,output_path=f"{RESULT_PATH}/{THIS_TIME}-segmented.gif", loop=0)
    dataset[0]['flair'].to_gif(axis=2, duration=10,output_path=f"{RESULT_PATH}/{THIS_TIME}-flair.gif", loop=0)
    dataset[0]['t1'].to_gif(axis=2, duration=10,output_path=f"{RESULT_PATH}/{THIS_TIME}-t1.gif", loop=0)
    dataset[0]['t2'].to_gif(axis=2, duration=10,output_path=f"{RESULT_PATH}/{THIS_TIME}-t2.gif", loop=0)
    dataset[0]['t1ce'].to_gif(axis=2, duration=10,output_path=f"{RESULT_PATH}/{THIS_TIME}-t1ce.gif", loop=0)

    os.remove(t1_path)
    os.remove(t2_path)
    os.remove(t1ce_path)
    os.remove(flair_path)

    return {
        "success" : True,
        "paths" : {
            "t1" : f"{THIS_TIME}-t1.gif",
            "t2" : f"{THIS_TIME}-t2.gif",
            "t1ce" : f"{THIS_TIME}-t1ce.gif",
            "flair" : f"{THIS_TIME}-flair.gif",
            "segmented" : f"{THIS_TIME}-segmented.gif",

        }
    }


            
app.mount("/", StaticFiles(directory="static",html = True), name="static")
app.mount("/results", StaticFiles(directory="static/results",html = True), name="static")