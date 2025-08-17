from fpdf import FPDF, HTMLMixin
import os
import jinja2
from datetime import datetime
import subprocess
import tempfile
import shutil

from .config import settings


REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


class _PDF(FPDF, HTMLMixin):
    """Extension that supports simple HTML rendering."""
    pass


class PDFReport:
    def __init__(
        self,
        title: str = "Relatório Técnico - Análise NIR",
        template_path: str | None = None,
        engine: str = "fpdf",
    ) -> None:
        """Initialize the PDF report."""
        self.title = title
        default_template = os.path.join(
            os.path.dirname(__file__), "templates", "report_template.html"
        )
        self.template_path = template_path or default_template
        self.engine = engine
        self.pdf = _PDF() if engine == "fpdf" else None
        self._html = ""

    def add_metrics(
        self,
        metrics: dict,
        params: dict | None = None,
        scatter_path: str = "",
        vip_path: str = "",
        conf_path: str = "",
        class_report_path: str = "",
        user: str = "",
        top_vips: list | None = None,
        range_used: str = "",
        interpretacao_vips: list | None = None,
        resumo_interpretativo: str | None = None,
        opt_results: list | None = None,
        opt_plot_path: str | None = None,
        result: dict | None = None,
    ) -> None:
        """Prepare HTML with metrics and optionally render using FPDF."""
        params = params or {}
        result = result or {}
        scatter_valid = (
            scatter_path if scatter_path and os.path.exists(scatter_path) else ""
        )
        vip_valid = vip_path if vip_path and os.path.exists(vip_path) else ""
        conf_valid = conf_path if conf_path and os.path.exists(conf_path) else ""
        class_valid = (
            class_report_path
            if class_report_path and os.path.exists(class_report_path)
            else ""
        )
        context = {
            "title": self.title,
            "metrics": metrics,
            "params": params,
            "logo": settings.assets_logo,
            "scatter_path": scatter_valid,
            "vip_path": vip_valid,
            "user": user or params.get("user", ""),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "preprocess_list": params.get("preprocess_list", []),
            "top_vips": top_vips,
            "range_used": result.get("range_used", range_used),
            "conf_path": conf_valid,
            "class_report_path": class_valid,
            "interpretacao_vips": interpretacao_vips,
            "resumo_interpretativo": resumo_interpretativo,
            "opt_results": opt_results or [],
            "opt_plot": (
                opt_plot_path if opt_plot_path and os.path.exists(opt_plot_path) else "",
            ),
            "validation_used": result.get("validation_used"),
            "n_splits_effective": result.get("n_splits_effective"),
            "best": result.get("best"),
            "per_class": result.get("per_class"),
        }

        curves = result.get("curves")
        curves_plot = ""
        curves_title = ""
        if curves:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ylabel = None
                for curve in curves:
                    pts = curve.get("points", [])
                    if not pts:
                        continue
                    xs = [p["n_components"] for p in pts]
                    ys_key = "MacroF1" if "MacroF1" in pts[0] else "RMSECV"
                    ys = [p.get(ys_key) for p in pts]
                    ylabel = ys_key
                    ax.plot(xs, ys, marker="o", label=curve.get("preprocess"))
                ax.set_xlabel("VL")
                if ylabel:
                    ax.set_ylabel(ylabel)
                ax.legend()
                title = 'Variáveis Latentes × Accuracy/MacroF1 (Classificação)' if ylabel == 'MacroF1' else 'Variáveis Latentes × RMSECV/R²CV (Regressão)'
                ax.set_title(title)
                fig.tight_layout()
                os.makedirs(REPORT_DIR, exist_ok=True)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=REPORT_DIR)
                fig.savefig(tmp.name)
                plt.close(fig)
                curves_plot = tmp.name
                curves_title = title
            except Exception:
                curves_plot = ""
                curves_title = ""
        context["curves_plot"] = curves_plot
        context["curves_title"] = curves_title

        if os.path.exists(self.template_path):
            with open(self.template_path, "r", encoding="utf-8") as fh:
                template = jinja2.Template(fh.read())
            html = template.render(**context)
        else:
            rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items()
            )
            html = f"""<h1>{self.title}</h1><table>{rows}</table>"""

        self._html = html

        if self.engine == "fpdf" and self.pdf is not None:
            self.pdf.add_page()
            try:
                self.pdf.write_html(html)
            except AttributeError:
                self.pdf.set_font("Arial", size=12)
                for key, value in metrics.items():
                    self.pdf.cell(0, 10, f"{key}: {value}", ln=True)

    def output(self, path: str) -> None:
        """Save PDF to disk using the chosen engine."""
        if self.engine == "fpdf" and self.pdf is not None:
            self.pdf.output(path)
            return

        if not shutil.which("wkhtmltopdf"):
            raise RuntimeError(
                "wkhtmltopdf not found. Please install it to use this engine."
            )

        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=False, encoding="utf-8"
        ) as tmp_html:
            tmp_html.write(self._html)
            tmp_html_path = tmp_html.name

        try:
            subprocess.run(["wkhtmltopdf", tmp_html_path, path], check=True)
        finally:
            os.remove(tmp_html_path)
