from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoTokenizer
import uvicorn
import os
import base64
import io
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = None
tokenizer = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

async def load_model_async():
    global model, tokenizer
    
    logger.info("開始加載模型...")
    
    model_name = "OpenGVLab/InternVL2_5-26B-MPO"  
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"檢測到 {gpu_count} 個可用GPU")
        
        if gpu_count >= 2:
            device_map = "auto"  # 自動分配到可用GPU上
        else:
            device_map = 0  # 使用單GPU
    else:
        device_map = "cpu"
        logger.warning("未檢測到GPU，將使用CPU運行（不推薦）")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        
        model = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,  
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        )
        
        logger.info("模型加載完成")
    except Exception as e:
        logger.error(f"模型加載失敗: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_model_async()
    yield
    logger.info("正在關閉應用...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

try:
    app.mount("/", StaticFiles(directory=".", html=True), name="static")
except RuntimeError as e:
    logger.warning(f"無法掛載靜態文件目錄: {str(e)}")

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    message: str
    image: Optional[str] = None
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_message = request.message
        image_data = request.image
        history = request.history or []
        
        pixel_values = None
        if image_data:
            try:
                image_content = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                image = Image.open(io.BytesIO(image_content))
                
                filename = f"{uuid.uuid4()}.jpg"
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                image.save(image_path)
                logger.info(f"圖像已保存到: {image_path}")
                
                pixel_values = load_image(image_path, input_size=448, max_num=12)
                pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
                
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)  
                    # pass
                    
            except Exception as e:
                logger.error(f"圖像處理失敗: {str(e)}")
                raise HTTPException(status_code=400, detail=f"圖像處理失敗: {str(e)}")
        
        chat_history = []
        for i in range(0, len(history), 2):
            if i+1 < len(history):
                chat_history.append((history[i].content, history[i+1].content))
        
        generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        if pixel_values is not None:
            question = f"<image>\n{user_message}"
        else:
            question = user_message
        
        if chat_history:
            response, _ = model.chat(tokenizer, pixel_values, question, generation_config, 
                                history=chat_history, return_history=True)
        else:
            response, _ = model.chat(tokenizer, pixel_values, question, generation_config, 
                                history=None, return_history=True)
        
        return {"response": response}
    
    except Exception as e:
        logger.error(f"處理請求時出現錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"處理請求失敗: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, log_level="info")