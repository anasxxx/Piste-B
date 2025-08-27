# Piste B — Fashion3D FastAPI → TripoSR

Passerelle FastAPI qui reçoit une image 2D et génère un mesh 3D en s'appuyant sur **TripoSR**.
- `api/server.py` : API `/health`, `/generate`
- `config.yaml` : chemins et URL de l'API TripoSR
- `scripts/run_api.sh` : démarrage du serveur
- `requirements_api.txt` : dépendances minimales

L'API tente d'abord d'appeler l'API TripoSR locale (`http://127.0.0.1:8001`).  
Si elle n'est pas dispo, fallback en exécution locale via `~/TripoSR/run.py`.
