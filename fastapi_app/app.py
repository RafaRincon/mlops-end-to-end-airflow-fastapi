from fastapi import FastAPI, HTTPException,Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from typing import Optional, Any
import os
import threading
import time

import pandas as pd
from prometheus_client import start_http_server, Counter, Histogram

import mlflow
import mlflow.pyfunc
from typing import Union

import logging


# -----------------------------
# Prometheus metrics
# -----------------------------
REQUEST_COUNT = Counter("api_requests_total", "Total requests to /predict")
REQUEST_LATENCY = Histogram("api_request_duration_seconds", "Latency for /predict")

# --- logging (optional but useful) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("api")

# --- add these metrics ---
ERROR_COUNT = Counter("api_errors_total", "Total unhandled errors")
STATUS_COUNT = Counter("api_status_total", "HTTP status by path and method", ["path", "method", "status"])

# -----------------------------
# Config via environment vars
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

MODEL_URI = os.getenv("MODEL_URI", "models:/customers_models@Production")

# Optional periodic reload; 0 disables background polling
RELOAD_INTERVAL_SECS = int(os.getenv("RELOAD_INTERVAL_SECS", "0"))

# -----------------------------
# Input schema (your data fields)
# -----------------------------
class InputData(BaseModel):
    age: Union[int, float]
    monthly_income: Union[int, float]
    debt_to_income_ratio: Union[int, float]
    transactions_per_month: Union[int, float]
    complaints_last_6m: Union[int, float]

# Map MLflow schema dtypes to pandas dtypes
_MLFLOW_TO_PD = {
    "long": "int64",      # MLflow 'long' → pandas int64
    "integer": "int64",   # (if present)
    "double": "float64",  # MLflow 'double' → pandas float64
    "float": "float64",   # (if present)
    "string": "object",
    "boolean": "bool",
}



def _get_expected_dtypes_from_signature(model: mlflow.pyfunc.PyFuncModel) -> dict:
    """
    Return {col_name: pandas_dtype} from MLflow input schema.
    If no signature, return {} (no coercion applied).
    """
    try:
        sig = model.metadata.get_input_schema()  # mlflow.types.Schema
        expected = {}
        for col in sig.inputs:
            # col is a ColSpec with .name and .type
            mlflow_type = str(col.type).lower()
            expected[col.name] = _MLFLOW_TO_PD.get(mlflow_type, None)
        return expected
    except Exception:
        return {}

def _is_integer_like(x: Any, tol: float = 1e-9) -> bool:
    try:
        xf = float(x)
        return abs(xf - round(xf)) <= tol
    except Exception:
        return False

def _coerce_df_to_signature(df: pd.DataFrame, expected: dict) -> pd.DataFrame:
    """
    Coerce df columns to the expected pandas dtypes based on MLflow signature.
    If an 'int64' column has non-integer-like values (e.g., 40.7), raise HTTP 422.
    """
    if not expected:
        # No signature → do nothing
        return df

    # Ensure all expected columns exist in df (missing → 422)
    missing = [c for c in expected.keys() if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {missing}"
        )

    dfc = df.copy()
    for col, pd_dtype in expected.items():
        if pd_dtype is None:
            # Unknown type, skip
            continue

        if pd_dtype == "int64":
            # Validate integer-likeness first
            bad = []
            for v in dfc[col].tolist():
                if not _is_integer_like(v):
                    bad.append(v)
            if bad:
                raise HTTPException(
                    status_code=422,
                    detail=f"Column '{col}' expects integer values; got non-integer-like: {bad}"
                )
            # Safe cast to int64
            dfc[col] = pd.to_numeric(dfc[col], errors="raise").astype("int64")

        elif pd_dtype == "float64":
            dfc[col] = pd.to_numeric(dfc[col], errors="raise").astype("float64")

        elif pd_dtype == "bool":
            # basic coercion (optional: enforce strict true/false set)
            dfc[col] = dfc[col].astype("bool")

        elif pd_dtype == "object":
            dfc[col] = dfc[col].astype("object")

        else:
            # Fallback: try astype; if it fails, surface 422
            try:
                dfc[col] = dfc[col].astype(pd_dtype)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Column '{col}' cannot be cast to {pd_dtype}: {e}"
                )
    # Reorder to signature order when possible
    ordered_cols = [c for c in expected.keys()]
    dfc = dfc[ordered_cols]
    return dfc

# Optional: for batch prediction (enable later if needed)
# class BatchInput(BaseModel):
#     rows: list[InputData]

# -----------------------------
# App + model state
# -----------------------------
app = FastAPI(title="Customer Quality Inference")

@app.middleware("http")
async def status_metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception:
        ERROR_COUNT.inc()
        logger.exception("Unhandled exception")
        return JSONResponse({"detail": "Internal Server Error"}, status_code=500)
    finally:
        # If response exists record status, else assume 500
        status_code = locals().get("response").status_code if "response" in locals() else 500
        STATUS_COUNT.labels(path=str(request.url.path), method=request.method, status=str(status_code)).inc()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_model: Optional[mlflow.pyfunc.PyFuncModel] = None
_model_version: Optional[str] = None  # for visibility
_lock = threading.Lock()

# Column order must match what the model expects
COLUMNS = [
    "age",
    "monthly_income",
    "debt_to_income_ratio",
    "transactions_per_month",
    "complaints_last_6m",
]

def _load_model() -> None:
    """Load (or reload) the current alias-target model."""
    global _model, _model_version
    m = mlflow.pyfunc.load_model(MODEL_URI)
    # Try to resolve a human-friendly version for /health
    try:
        # If using alias like models:/name@champion, fetch where alias points now
        name = MODEL_URI.split("models:/", 1)[1].split("@", 1)[0].split("/", 1)[0]
        client = mlflow.MlflowClient()
        # If alias is present, this call will succeed
        alias = MODEL_URI.split("@", 1)[1] if "@" in MODEL_URI else None  # noqa: E999 (editor highlighter)
        if alias:
            alias_info = client.get_registered_model_alias(name, alias)
            version = alias_info.version
        else:
            # Fallback: if using explicit stage or version in URI, leave as None
            version = None
    except Exception:
        version = None

    with _lock:
        _model = m
        _model_version = str(version) if version is not None else "unknown"

def _background_reloader():
    """Poll for alias changes; reload when alias points to a new version."""
    while True:
        time.sleep(RELOAD_INTERVAL_SECS)
        try:
            # Resolve current alias target
            name = MODEL_URI.split("models:/", 1)[1].split("@", 1)[0].split("/", 1)[0]
            alias = MODEL_URI.split("@", 1)[1] if "@" in MODEL_URI else None  # noqa: E999
            if not alias:
                continue  # alias polling only

            client = mlflow.MlflowClient()
            info = client.get_registered_model_alias(name, alias)
            new_version = str(info.version)

            reload_needed = False
            with _lock:
                reload_needed = (_model_version != new_version)
            if reload_needed:
                _load_model()
                print(f"[reload] Switched alias {alias} → v{new_version}")
        except Exception as e:
            print(f"[reload] polling error: {e}")

# -----------------------------
# Lifecycle hooks
# -----------------------------
@app.on_event("startup")
def _startup():
    start_http_server(8001)  # Prometheus metrics on 8001
    _load_model()
    if RELOAD_INTERVAL_SECS > 0:
        t = threading.Thread(target=_background_reloader, daemon=True)
        t.start()

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "Prediction API is live"}

@app.get("/health")
def health():
    with _lock:
        v = _model_version
    return {
        "status": "ok",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MODEL_URI,
        "loaded_version": v,
    }

@app.post("/reload")
def reload_now():
    _load_model()
    with _lock:
        v = _model_version
    return {"reloaded_to": MODEL_URI, "version": v}

@app.post("/predict")
@REQUEST_LATENCY.time()
def predict(data: InputData):
    REQUEST_COUNT.inc()
    with _lock:
        if _model is None:
            # 503 -> counted in STATUS_COUNT via middleware
            raise HTTPException(status_code=503, detail="Model not loaded")

        row = {
            "age": data.age,
            "monthly_income": data.monthly_income,
            "debt_to_income_ratio": data.debt_to_income_ratio,
            "transactions_per_month": data.transactions_per_month,
            "complaints_last_6m": data.complaints_last_6m,
        }
        df = pd.DataFrame([row], columns=COLUMNS)
        expected = _get_expected_dtypes_from_signature(_model)
        df = _coerce_df_to_signature(df, expected)

        try:
            preds = _model.predict(df)
        except Exception:
            ERROR_COUNT.inc()
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed")

    pred0 = preds[0].item() if hasattr(preds[0], "item") else preds[0]
    return {"prediction": pred0}