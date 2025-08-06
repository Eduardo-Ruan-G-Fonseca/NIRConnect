import jinja2, os
from .config import settings

def render_html_report(metrics: dict, title="NIR Analysis Report") -> str:
    template_str = """<!DOCTYPE html>
<html>
<head>
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { color: #444; }
    table { border-collapse: collapse; width: 100%%; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <h2>Métricas</h2>
  <table>
    <tr><th>Métrica</th><th>Valor</th></tr>
    {% for k, v in metrics.items() %}
    <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
    {% endfor %}
  </table>
</body>
</html>"""
    template = jinja2.Template(template_str)
    return template.render(title=title, metrics=metrics)
