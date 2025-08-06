import os
import json
import jinja2
from core.config import settings, METRICS_FILE

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
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {"R2": 0.0, "RMSE": 0.0, "Accuracy": 0.0}

def generate_dashboard(output_path='dashboard.html'):
    log_file = None
    for f in os.listdir(settings.logging_dir):
        if f.endswith('.log'):
            log_file = os.path.join(settings.logging_dir, f)
            break

    logs = ''
    if log_file:
        with open(log_file, 'r') as lf:
            logs = lf.read()

    chart_data = parse_log_levels(logs)
    chart_data_json = json.dumps(chart_data)
    model_data_json = json.dumps(get_model_statistics())

    env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates"))
    template = env.get_template("dashboard.html")
    html = template.render(
        logs=logs,
        chart_data=chart_data_json,
        model_data=model_data_json,
    )

    with open(output_path, 'w') as out:
        out.write(html)
    print(f'Dashboard gerado: {output_path}')

if __name__ == '__main__':
    generate_dashboard()
