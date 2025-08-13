# NIRConnect

Este repositório está organizado em duas partes:

## Backend
Código FastAPI localizado em `backend/`.

Para executar (via PyCharm ou terminal):
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

 Frontend
Aplicação React localizada em frontend/.

Para executar (via VS Code ou terminal):
```bash
cd frontend
npm install # na primeira execução
npm run dev
```
## API de PLS

As rotas `/preprocess`, `/train` e `/predict` permitem sanear dados, treinar o modelo PLS com validação cruzada e gerar previsões usando o pipeline blindado contra NaN/Inf.

