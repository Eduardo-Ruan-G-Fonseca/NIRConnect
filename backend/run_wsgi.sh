#!/bin/bash

# Ativa o virtualenv, se existir
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Inicia o servidor Gunicorn com 2 workers, escutando em 0.0.0.0:5000
gunicorn -w 2 -b 0.0.0.0:5000 public_html.api.main:app
