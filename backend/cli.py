import typer
import pandas as pd
from .core.pls import train_pls
from .core.bootstrap import bootstrap_metrics
from .core.report_pdf import PDFReport
from .generate_dashboard import generate_dashboard
import uvicorn
import json

app = typer.Typer(help="CLI para Análise NIR.")


@app.command()
def train(
    data_path: str,
    target: str,
    n_components: int = 5,
    classification: bool = False,
    n_bootstrap: int = 0,
):
    df = (
        pd.read_csv(data_path)
        if data_path.endswith(".csv")
        else pd.read_excel(data_path)
    )
    X = df.drop(columns=[target]).values
    y = df[target].values
    model, metrics, extra = train_pls(
        X, y, n_components=n_components, classification=classification
    )
    if n_bootstrap > 0:
        boot = bootstrap_metrics(
            X,
            y,
            n_components=n_components,
            classification=classification,
            n_bootstrap=n_bootstrap,
        )
        metrics["bootstrap"] = boot
    typer.echo(json.dumps({"metrics": metrics, "vip": extra["vip"]}, indent=2))


@app.command()
def report(metrics_json: str, output: str = "report.pdf", engine: str = "fpdf"):
    """Gera relatório em PDF a partir de métricas JSON."""
    metrics = json.loads(metrics_json)
    pdf = PDFReport(engine=engine)
    pdf.add_metrics(metrics)
    pdf.output(output)
    typer.echo(f"Relatório salvo em {output}")


@app.command()
def html_to_pdf(html_file: str, output: str):
    """Converte um arquivo HTML em PDF utilizando wkhtmltopdf."""
    import shutil
    import subprocess

    if not shutil.which("wkhtmltopdf"):
        typer.echo("Erro: wkhtmltopdf não encontrado.", err=True)
        raise typer.Exit(code=1)
    try:
        subprocess.run(["wkhtmltopdf", html_file, output], check=True)
        typer.echo(f"PDF salvo em {output}")
    except subprocess.CalledProcessError:
        typer.echo("Falha ao gerar PDF", err=True)


@app.command()
def dashboard(output: str = "dashboard.html"):
    """Gera o dashboard de logs e métricas."""
    generate_dashboard(output)
    typer.echo(f"Dashboard gerado em {output}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Inicia a API FastAPI."""
    uvicorn.run("api_fastapi.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
