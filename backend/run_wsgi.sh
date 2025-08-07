#!/bin/bash
if [ -d "venv" ]; then
  source venv/bin/activate
fi
gunicorn -w 2 -b 0.0.0.0:5000 main:app
