# server.py  —  Driver Drowsiness Detection  (MongoDB edition)
# ─────────────────────────────────────────────────────────────
# Install:
#   pip install fastapi uvicorn motor pymongo opencv-python mediapipe numpy python-multipart
#
# Run:
#   uvicorn server:app --reload
#
# MongoDB:  default  mongodb://localhost:27017
#           override via env: MONGO_URL=mongodb+srv://...

import base64
import datetime
import logging
import math
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from bson import ObjectId
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

# ── logging ───────────────────────────────────────────────────────────────────
LOG_FILE = "drowsiness_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("drowsiness")

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
_client   = AsyncIOMotorClient(MONGODB_URI)
db        = _client["drowsiness_db"]

col_drivers  = db["drivers"]   # driver profiles + photo (base-64)
col_sessions = db["sessions"]  # one doc per driving session
col_alerts   = db["alerts"]    # individual warning/danger events

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DrowseGuard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://drowse-guard-driver-drowsiness-detection-system-fstacdabj.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MediaPipe Face Landmarker (Tasks API) ─────────────────────────────────────
# Newer MediaPipe releases no longer expose the legacy mp.solutions API.
# The Tasks API is used here so the backend works with current MediaPipe builds.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path("/tmp/face_landmarker.task")


def _ensure_face_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    log.info("Downloading MediaPipe face_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info("MediaPipe face landmarker model ready.")


_ensure_face_model()

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_mp_base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
_mp_options = mp_vision.FaceLandmarkerOptions(
    base_options=_mp_base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
_face_landmarker = mp_vision.FaceLandmarker.create_from_options(_mp_options)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 13, 14]
HEAD_IDS  = [1, 152, 263, 33, 287, 57]

FACE_3D = np.array([
    (  0.0,    0.0,    0.0),
    (  0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)

EAR_THRESH   = 0.25
MAR_THRESH   = 0.55
PITCH_THRESH = 15.0

# ── Active session state (resets on server restart) ───────────────────────────
_active: dict = {}   # {"session_id": str, "driver_id": str}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _px(lm, i, w, h):
    return int(lm[i].x * w), int(lm[i].y * h)

def _ear(lm, ids, w, h):
    p = [_px(lm, i, w, h) for i in ids]
    v = math.dist(p[1], p[5]) + math.dist(p[2], p[4])
    hz = math.dist(p[0], p[3])
    return v / (2 * hz) if hz else 0.0

def _mar(lm, ids, w, h):
    p = [_px(lm, i, w, h) for i in ids]
    return math.dist(p[2], p[3]) / math.dist(p[0], p[1]) if math.dist(p[0], p[1]) else 0.0

def _pitch(lm, w, h):
    pts2d = np.array([(lm[i].x * w, lm[i].y * h) for i in HEAD_IDS], dtype=np.float64)
    cam   = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rv, tv = cv2.solvePnP(FACE_3D, pts2d, cam, np.zeros((4,1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0
    rmat, _ = cv2.Rodrigues(rv)
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, tv)))
    return float(angles[0])

def _decide(eyes, yawn, nod):
    n = sum([eyes, yawn, nod])
    if n >= 2: return "danger",  "⚠️ DANGER — pull over and rest now!"
    if n == 1: return "warning", "😴 Warning — you look drowsy."
    return "safe", "✅ Driver is alert."

def _serial(doc):
    """Make a MongoDB doc JSON-safe."""
    if doc is None:
        return None
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime.datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out

def _empty(face=False):
    return {
        "face_found": face, "ear": 0.0, "mar": 0.0, "pitch": 0.0,
        "eyes_closed": False, "yawning": False, "head_nodding": False,
        "alert_level": "safe",
        "message": "No face detected — make sure your face is visible.",
    }

# ── Static ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("index.html")

# ═════════════════════════════════════════════════════════════════════════════
# DRIVERS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/drivers")
async def create_driver(
    name:           str        = Form(...),
    license_number: str        = Form(""),
    vehicle:        str        = Form(""),
    phone:          str        = Form(""),
    photo:          UploadFile = File(None),
):
    photo_b64 = None
    if photo and photo.filename:
        raw  = await photo.read()
        ext  = photo.filename.rsplit(".", 1)[-1].lower()
        mime = f"image/{ext}" if ext in ("jpg","jpeg","png","webp","gif") else "image/jpeg"
        photo_b64 = f"data:{mime};base64,{base64.b64encode(raw).decode()}"

    doc = {
        "name":           name,
        "license_number": license_number,
        "vehicle":        vehicle,
        "phone":          phone,
        "photo":          photo_b64,
        "created_at":     datetime.datetime.utcnow(),
        "total_sessions": 0,
        "total_warnings": 0,
        "total_dangers":  0,
    }
    res = await col_drivers.insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    log.info(f"Driver created: {name}")
    return _serial(doc)


@app.get("/drivers")
async def list_drivers():
    return [_serial(d) async for d in col_drivers.find().sort("created_at", -1)]


@app.get("/drivers/{did}")
async def get_driver(did: str):
    return _serial(await col_drivers.find_one({"_id": ObjectId(did)}))


@app.put("/drivers/{did}")
async def update_driver(
    did:            str,
    name:           str        = Form(...),
    license_number: str        = Form(""),
    vehicle:        str        = Form(""),
    phone:          str        = Form(""),
    photo:          UploadFile = File(None),
):
    upd = {"name": name, "license_number": license_number,
           "vehicle": vehicle, "phone": phone}
    if photo and photo.filename:
        raw  = await photo.read()
        ext  = photo.filename.rsplit(".", 1)[-1].lower()
        mime = f"image/{ext}" if ext in ("jpg","jpeg","png","webp","gif") else "image/jpeg"
        upd["photo"] = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    await col_drivers.update_one({"_id": ObjectId(did)}, {"$set": upd})
    return _serial(await col_drivers.find_one({"_id": ObjectId(did)}))


@app.delete("/drivers/{did}")
async def delete_driver(did: str):
    await col_drivers.delete_one({"_id": ObjectId(did)})
    await col_sessions.delete_many({"driver_id": did})
    await col_alerts.delete_many({"driver_id": did})
    if _active.get("driver_id") == did:
        _active.clear()
    return {"deleted": did}

# ═════════════════════════════════════════════════════════════════════════════
# SESSIONS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/sessions/start")
async def start_session(body: dict):
    did = body.get("driver_id", "")
    if not did:
        return {"error": "driver_id required"}

    # close any open session first
    if _active.get("session_id"):
        await _close(_active["session_id"])

    doc = {
        "driver_id":  did,
        "started_at": datetime.datetime.utcnow(),
        "ended_at":   None,
        "warnings":   0,
        "dangers":    0,
        "total_alerts": 0,
    }
    res = await col_sessions.insert_one(doc)
    sid = str(res.inserted_id)
    _active["session_id"] = sid
    _active["driver_id"]  = did
    await col_drivers.update_one({"_id": ObjectId(did)}, {"$inc": {"total_sessions": 1}})
    log.info(f"Session started: {sid}  driver: {did}")
    return {"session_id": sid}


@app.post("/sessions/end")
async def end_session():
    if not _active.get("session_id"):
        return {"error": "no active session"}
    sid = _active.pop("session_id")
    _active.pop("driver_id", None)
    await _close(sid)
    return {"ended": sid}


async def _close(sid: str):
    await col_sessions.update_one(
        {"_id": ObjectId(sid)},
        {"$set": {"ended_at": datetime.datetime.utcnow()}}
    )


@app.get("/sessions/active")
async def active_session():
    if not _active.get("session_id"):
        return {"active": False}
    s = _serial(await col_sessions.find_one({"_id": ObjectId(_active["session_id"])}))
    d = _serial(await col_drivers.find_one({"_id": ObjectId(_active["driver_id"])}))
    return {"active": True, "session": s, "driver": d}


@app.get("/sessions/driver/{did}")
async def sessions_for_driver(did: str):
    out = []
    async for s in col_sessions.find({"driver_id": did}).sort("started_at", -1).limit(25):
        s = _serial(s)
        s["events"] = [
            _serial(a) async for a in
            col_alerts.find({"session_id": s["_id"]}).sort("timestamp", 1)
        ]
        out.append(s)
    return out


@app.get("/alerts/live")
async def live_alerts():
    if not _active.get("session_id"):
        return []
    return [
        _serial(a) async for a in
        col_alerts.find({"session_id": _active["session_id"]}).sort("timestamp", -1).limit(50)
    ]

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSE  (called every 500 ms by the browser)
# ═════════════════════════════════════════════════════════════════════════════

class Frame(BaseModel):
    image: str  # "data:image/jpeg;base64,..."

@app.post("/analyse")
async def analyse(frame: Frame):
    try:
        raw   = base64.b64decode(frame.image.split(",")[-1])
        arr   = np.frombuffer(raw, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        log.error(f"decode error: {exc}")
        return _empty()

    if image is None:
        return _empty()

    h, w = image.shape[:2]

    # MediaPipe Tasks expects an RGB mp.Image.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return _empty(face=False)

    lm = result.face_landmarks[0]
    ear   = (_ear(lm, LEFT_EYE, w, h) + _ear(lm, RIGHT_EYE, w, h)) / 2
    mar   = _mar(lm, MOUTH, w, h)
    pitch = _pitch(lm, w, h)

    eyes  = ear        < EAR_THRESH
    yawn  = mar        > MAR_THRESH
    nod   = abs(pitch) > PITCH_THRESH
    level, msg = _decide(eyes, yawn, nod)

    result_dict = {
        "face_found":   True,
        "ear":          round(ear,   3),
        "mar":          round(mar,   3),
        "pitch":        round(pitch, 1),
        "eyes_closed":  eyes,
        "yawning":      yawn,
        "head_nodding": nod,
        "alert_level":  level,
        "message":      msg,
    }

    # Persist warning / danger events to MongoDB
    if _active.get("session_id") and level != "safe":
        alert = {
            "session_id":  _active["session_id"],
            "driver_id":   _active["driver_id"],
            "timestamp":   datetime.datetime.utcnow(),
            "alert_level": level,
            "message":     msg,
            "ear":         result_dict["ear"],
            "mar":         result_dict["mar"],
            "pitch":       result_dict["pitch"],
            "eyes_closed": eyes,
            "yawning":     yawn,
            "head_nodding": nod,
        }
        await col_alerts.insert_one(alert)
        await col_sessions.update_one(
            {"_id": ObjectId(_active["session_id"])},
            {"$inc": {"total_alerts": 1,
                      "warnings": (1 if level == "warning" else 0),
                      "dangers":  (1 if level == "danger"  else 0)}},
        )
        field = "total_warnings" if level == "warning" else "total_dangers"
        await col_drivers.update_one(
            {"_id": ObjectId(_active["driver_id"])}, {"$inc": {field: 1}}
        )
        log.warning(f"[{level.upper()}] EAR={ear:.3f} MAR={mar:.3f} Pitch={pitch:.1f}°")

    return result_dict

# ═════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    await col_drivers.create_index("created_at")
    await col_sessions.create_index("driver_id")
    await col_alerts.create_index([("session_id", 1), ("timestamp", -1)])
    log.info("=" * 52)
    log.info("DrowseGuard server started")
    log.info(f"MongoDB : {MONGODB_URI}")
    log.info("UI      : http://localhost:8000")
    log.info("=" * 52)

@app.on_event("shutdown")
async def shutdown():
    if _active.get("session_id"):
        await _close(_active["session_id"])
    _client.close()
    log.info("Server stopped.")
