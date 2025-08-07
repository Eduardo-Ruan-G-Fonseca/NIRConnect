#!/bin/bash
echo "Monitorando API..."
curl -s http://127.0.0.1:5000/health || echo "API offline"
