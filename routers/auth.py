"""
Router de autenticación — JWT sin base de datos.

Endpoint: POST /api/auth/login
- Verifica usuario y contraseña contra hashes bcrypt en variables de entorno
- Devuelve JWT firmado con expiración de 8 horas
"""

import os
import time

import bcrypt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from jose import jwt

router = APIRouter()

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_H  = 8


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if not secret:
        raise RuntimeError("JWT_SECRET no está configurado en el archivo .env")
    return secret


def _get_users() -> dict:
    """Lee usuarios desde variables de entorno en cada request (respeta carga de .env)."""
    return {
        os.getenv("ADMIN_USER", "admin"): {
            "hash":    os.getenv("ADMIN_HASH", ""),
            "role":    "admin",
            "display": "Administrador",
        },
        os.getenv("VIEWER_USER", "visualizador"): {
            "hash":    os.getenv("VIEWER_HASH", ""),
            "role":    "viewer",
            "display": "Visualizador",
        },
    }


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
async def login(req: LoginRequest) -> dict:
    """Autentica usuario y devuelve JWT firmado."""
    users = _get_users()
    user  = users.get(req.username.strip())

    # Mismo mensaje para usuario inválido o contraseña incorrecta (evita enumeración)
    if not user or not user["hash"]:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    if not _verify_password(req.password, user["hash"]):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    exp = int(time.time()) + JWT_EXPIRE_H * 3600
    payload = {
        "sub":     req.username,
        "role":    user["role"],
        "display": user["display"],
        "exp":     exp,
    }
    token = jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)

    return {
        "token":   token,
        "user":    user["display"],
        "role":    user["role"],
        "expires": exp,
    }
