#!/bin/bash
# Conversão HTML -> PDF usando wkhtmltopdf (se instalado)
if [ $# -lt 2 ]; then
  echo "Uso: $0 arquivo.html saida.pdf"
  exit 1
fi

# Ensure wkhtmltopdf is available
if ! command -v wkhtmltopdf >/dev/null 2>&1; then
  echo "Erro: wkhtmltopdf n\u00e3o encontrado. Instale-o para usar este script." >&2
  exit 1
fi

wkhtmltopdf "$1" "$2"
