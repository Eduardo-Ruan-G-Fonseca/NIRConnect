import os
import json
from .core.config import settings, METRICS_FILE

def parse_log_levels(log_content):
    levels = ["INFO", "ERROR", "WARNING", "DEBUG"]
    counts = {lvl: 0 for lvl in levels}
    for line in log_content.splitlines():
        for lvl in levels:
            if lvl in line:
                counts[lvl] += 1
    return counts

def get_model_statistics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"R2": 0.0, "RMSE": 0.0, "Accuracy": 0.0}

def generate_dashboard(output_path: str = 'dashboard.html'):
    """Gera um dashboard simples em HTML sem uso de templates."""
    log_file = None
    for f in os.listdir(settings.logging_dir):
        if f.endswith('.log'):
            log_file = os.path.join(settings.logging_dir, f)
            break

    logs = ''
    if log_file:
        with open(log_file, 'r', encoding='utf-8') as lf:
            logs = lf.read()

    chart_data = json.dumps(parse_log_levels(logs))
    model_data = json.dumps(get_model_statistics(), indent=2)
    html = f"""
    <html><head><title>Dashboard</title></head>
    <body>
    <h1>Logs</h1>
    <pre>{logs}</pre>
    <h2>Contagens</h2>
    <pre>{chart_data}</pre>
    <h2>Métricas</h2>
    <pre>{model_data}</pre>
    </body></html>
    """
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(html)
    print(f'Dashboard gerado: {output_path}')

if __name__ == '__main__':
    generate_dashboard()
