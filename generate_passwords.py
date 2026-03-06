#!/usr/bin/env python3
"""
Genera hashes bcrypt para las contraseñas del sistema MBC.
Copia los valores generados al archivo .env

Uso:
    python generate_passwords.py
"""

import bcrypt

print("=" * 55)
print("  Generador de contraseñas — Sistema MBC")
print("=" * 55)
print()

admin_pass  = input("Contraseña para ADMIN        : ")
viewer_pass = input("Contraseña para VISUALIZADOR : ")

admin_hash  = bcrypt.hashpw(admin_pass.encode("utf-8"),  bcrypt.gensalt()).decode("utf-8")
viewer_hash = bcrypt.hashpw(viewer_pass.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

print()
print("Copia estas líneas en tu archivo .env:")
print("-" * 55)
print(f"ADMIN_HASH={admin_hash}")
print(f"VIEWER_HASH={viewer_hash}")
print("-" * 55)
