from fpdf import FPDF, HTMLMixin
import os
import math
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
        train_metrics: dict | None = None,
        validation_metrics: dict | None = None,
        vip_data: list | dict | None = None,
        confusion_matrix: dict | None = None,
        residuals: dict | None = None,
        influence: dict | None = None,
        distributions: dict | None = None,
        predictions: dict | None = None,
        latent: dict | None = None,
        task: str | None = None,
        user_inputs: dict | None = None,
    ) -> None:
        """Prepare HTML with metrics and optionally render using FPDF."""

        params = params or {}
        result = result or {}
        user_inputs = user_inputs or params
        train_metrics = train_metrics or {}
        validation_metrics = validation_metrics or {}
        vip_data = vip_data or result.get("vip") or []
        confusion_matrix = confusion_matrix or result.get("confusion_matrix")
        residuals = residuals or result.get("residuals")
        influence = influence or result.get("influence")
        distributions = distributions or result.get("distributions")
        predictions = predictions or result.get("predictions")
        latent = latent or result.get("latent")

        resolved_task = (
            task
            or result.get("task")
            or params.get("task")
            or ("classification" if params.get("classification") else "regression")
        )

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

        def safe_float(value):
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num):
                return None
            return num

        def format_number(value, decimals: int = 4):
            num = safe_float(value)
            if num is None:
                return ""
            if abs(num - round(num)) < 1e-6:
                return str(int(round(num)))
            return f"{num:.{decimals}f}"

        def format_value(value):
            if value is None:
                return ""
            if isinstance(value, bool):
                return "Sim" if value else "Não"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted = format_number(value)
                return formatted or ""
            if isinstance(value, (list, tuple, set)):
                parts = [format_value(v) for v in value if format_value(v)]
                return ", ".join(parts)
            if isinstance(value, dict):
                parts = []
                for key, val in value.items():
                    formatted = format_value(val)
                    if formatted:
                        parts.append(f"{key}: {formatted}")
                return "; ".join(parts)
            return str(value)

        def format_range(value):
            if value is None:
                return ""
            low = None
            high = None
            if isinstance(value, dict):
                low = value.get("min") or value.get(0)
                high = value.get("max") or value.get(1)
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                low, high = value[0], value[1]
            else:
                return format_value(value)
            low_fmt = format_number(low, 2)
            high_fmt = format_number(high, 2)
            if low_fmt and high_fmt:
                return f"{low_fmt} – {high_fmt} nm"
            return format_value(value)

        def metrics_to_rows(data):
            rows: list[dict[str, str]] = []
            if isinstance(data, dict):
                for name, value in data.items():
                    if value is None:
                        continue
                    rows.append({"name": str(name), "value": format_value(value)})
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        name = entry.get("metric") or entry.get("name")
                        value = entry.get("value") or entry.get("score")
                        if name is not None:
                            rows.append({"name": str(name), "value": format_value(value)})
            return rows

        summary_rows: list[dict[str, str]] = []
        dataset_id = user_inputs.get("dataset_id") or user_inputs.get("datasetId")
        if dataset_id:
            summary_rows.append({"label": "Dataset", "value": str(dataset_id)})
        target = user_inputs.get("target") or user_inputs.get("target_name")
        if target:
            summary_rows.append({"label": "Variável resposta", "value": str(target)})
        summary_rows.append(
            {
                "label": "Tarefa",
                "value": "Classificação"
                if resolved_task == "classification"
                else "Regressão",
            }
        )
        validation_method = (
            user_inputs.get("validation_method")
            or result.get("validation_used")
            or ""
        )
        if validation_method:
            summary_rows.append(
                {"label": "Validação", "value": str(validation_method)}
            )
        n_splits = (
            user_inputs.get("n_splits")
            or user_inputs.get("folds")
            or result.get("n_splits_effective")
        )
        if n_splits:
            summary_rows.append(
                {"label": "Splits", "value": format_value(n_splits)}
            )
        n_bootstrap = user_inputs.get("n_bootstrap")
        if n_bootstrap:
            summary_rows.append(
                {
                    "label": "Bootstrap",
                    "value": format_value(n_bootstrap),
                }
            )
        threshold = user_inputs.get("threshold")
        if threshold is not None:
            summary_rows.append(
                {
                    "label": "Limiar de decisão",
                    "value": format_value(threshold),
                }
            )
        n_components = (
            user_inputs.get("n_components")
            or result.get("best", {}).get("n_components")
        )
        if n_components:
            summary_rows.append(
                {
                    "label": "Variáveis latentes",
                    "value": format_value(n_components),
                }
            )
        min_score = user_inputs.get("min_score")
        if min_score is not None:
            summary_rows.append(
                {"label": "Score mínimo", "value": format_value(min_score)}
            )
        best_score = user_inputs.get("best_score") or result.get("best", {}).get(
            "score"
        )
        if best_score is not None:
            summary_rows.append(
                {"label": "Score otimizado", "value": format_value(best_score)}
            )
        optimized = user_inputs.get("optimized")
        if optimized is not None:
            summary_rows.append(
                {"label": "Otimização realizada", "value": format_value(optimized)}
            )

        range_source = (
            result.get("range_used")
            or user_inputs.get("range_used")
            or range_used
            or user_inputs.get("spectral_range")
        )
        if range_source:
            summary_rows.append(
                {"label": "Faixa espectral", "value": format_range(range_source)}
            )

        validation_params = user_inputs.get("validation_params")
        validation_rows = []
        if isinstance(validation_params, dict):
            for key, value in validation_params.items():
                formatted = format_value(value)
                if formatted:
                    validation_rows.append(
                        {"label": f"Validação · {key}", "value": formatted}
                    )

        preprocess_list: list[str] = []

        def register_steps(source):
            if isinstance(source, list):
                for step in source:
                    if isinstance(step, dict):
                        name = step.get("method") or step.get("name")
                    else:
                        name = step
                    if not name:
                        continue
                    name_str = str(name)
                    if name_str not in preprocess_list:
                        preprocess_list.append(name_str)

        register_steps(user_inputs.get("preprocess_steps"))
        register_steps(user_inputs.get("preprocess"))
        register_steps(
            user_inputs.get("best_params", {}).get("preprocess")
            if isinstance(user_inputs.get("best_params"), dict)
            else None
        )
        register_steps(result.get("best", {}).get("preprocess"))

        preprocess_grid_rows: list[str] = []
        grid_source = user_inputs.get("preprocess_grid") or result.get("preprocess_grid")
        if isinstance(grid_source, list):
            for idx, pipeline in enumerate(grid_source, start=1):
                steps: list[str] = []
                if isinstance(pipeline, (list, tuple)):
                    for step in pipeline:
                        if isinstance(step, dict):
                            name = step.get("method") or step.get("name")
                        else:
                            name = step
                        if name:
                            steps.append(str(name))
                elif pipeline:
                    steps.append(str(pipeline))
                text = " + ".join(steps) if steps else "Sem pré-processamento"
                preprocess_grid_rows.append(f"{idx}. {text}")

        def normalize_sg(value):
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                parsed = [safe_float(v) for v in value[:3]]
                if all(v is not None for v in parsed):
                    return [int(round(v)) for v in parsed]
            if isinstance(value, dict):
                window = value.get("window") or value.get("window_length") or value.get(0)
                poly = value.get("poly") or value.get("polyorder") or value.get(1)
                deriv = value.get("deriv") or value.get("derivative") or value.get(2)
                return normalize_sg([window, poly, deriv])
            return None

        sg_params = None
        if user_inputs.get("sg") is not None:
            sg_params = normalize_sg(user_inputs.get("sg"))
        elif isinstance(user_inputs.get("sg_params"), list) and user_inputs["sg_params"]:
            sg_params = normalize_sg(user_inputs["sg_params"][0])

        if sg_params is None and isinstance(preprocess_list, list):
            if any(step.lower().startswith("sg") for step in preprocess_list):
                sg_params = [11, 2, 1]

        sg_description = None
        if sg_params:
            sg_description = (
                f"Savitzky-Golay (janela {sg_params[0]}, ordem {sg_params[1]}, derivada {sg_params[2]})"
            )

        per_class_rows: list[dict[str, str]] = []
        per_class_data = result.get("per_class")
        if isinstance(per_class_data, dict):
            for cls, values in per_class_data.items():
                if isinstance(values, dict):
                    per_class_rows.append(
                        {
                            "label": str(cls),
                            "precision": format_number(
                                values.get("precision") or values.get("prec"), 3
                            ),
                            "recall": format_number(
                                values.get("recall") or values.get("sens"), 3
                            ),
                            "f1": format_number(
                                values.get("f1") or values.get("f1_score"), 3
                            ),
                            "support": format_value(values.get("support")),
                        }
                    )
        elif isinstance(per_class_data, list):
            for row in per_class_data:
                if isinstance(row, dict):
                    per_class_rows.append(
                        {
                            "label": str(
                                row.get("label")
                                or row.get("classe")
                                or row.get("class")
                                or row.get("Classe")
                            ),
                            "precision": format_number(row.get("precision"), 3),
                            "recall": format_number(row.get("recall"), 3),
                            "f1": format_number(row.get("f1"), 3),
                            "support": format_value(row.get("support")),
                        }
                    )

        best_raw = result.get("best")
        best_copy: dict | None = None
        best_metrics_rows: list[dict[str, str]] = []
        if isinstance(best_raw, dict) and best_raw:
            best_copy = dict(best_raw)
            preprocess_value = best_copy.get("preprocess")
            if isinstance(preprocess_value, (list, tuple)):
                steps = [str(step) for step in preprocess_value if step]
                best_copy["preprocess"] = (
                    " + ".join(steps) if steps else "Sem pré-processamento"
                )
            elif preprocess_value is None:
                best_copy["preprocess"] = "Sem pré-processamento"

            raw_val_metrics = (
                best_copy.get("val_metrics")
                or best_copy.get("metrics")
                or metrics
            )
            best_metrics_rows = metrics_to_rows(raw_val_metrics)
            best_copy["val_metrics"] = best_metrics_rows
            if best_copy.get("score") is not None:
                best_copy["score"] = format_value(best_copy.get("score"))
            if best_copy.get("threshold") is not None:
                best_copy["threshold"] = format_value(best_copy.get("threshold"))

        metrics_rows = metrics_to_rows(metrics)
        train_metrics_rows = metrics_to_rows(train_metrics)
        validation_metrics_rows = metrics_to_rows(validation_metrics)

        def create_plot(build_func):
            try:
                import matplotlib.pyplot as plt
            except Exception:
                return ""
            fig = build_func(plt)
            if fig is None:
                return ""
            os.makedirs(REPORT_DIR, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=REPORT_DIR)
            fig.savefig(tmp.name, dpi=180, bbox_inches="tight")
            plt.close(fig)
            return tmp.name

        curves_plot = ""
        curves_title = ""
        curves = result.get("curves")
        if curves:
            def build_curves(plt):
                fig, ax = plt.subplots()
                ylabel = None
                legend_labels = []
                for curve in curves:
                    if not isinstance(curve, dict):
                        continue
                    pts = curve.get("points", [])
                    if not pts:
                        continue
                    xs = [p.get("n_components") for p in pts if p.get("n_components") is not None]
                    if not xs:
                        continue
                    metric_key = None
                    if "MacroF1" in pts[0]:
                        metric_key = "MacroF1"
                    elif "RMSECV" in pts[0]:
                        metric_key = "RMSECV"
                    elif "R2CV" in pts[0]:
                        metric_key = "R2CV"
                    ys = [p.get(metric_key) for p in pts] if metric_key else []
                    if not ys:
                        continue
                    ylabel_local = metric_key or "Score"
                    ylabel = ylabel or ylabel_local
                    legend_labels.append(curve.get("preprocess") or curve.get("label"))
                    ax.plot(xs, ys, marker="o", label=legend_labels[-1])
                ax.set_xlabel("Variáveis latentes")
                if ylabel:
                    ax.set_ylabel(ylabel)
                if legend_labels:
                    ax.legend(loc="best")
                ax.grid(True, linestyle="--", alpha=0.3)
                title = (
                    "VL × Métrica de Validação (Classificação)"
                    if ylabel and "Macro" in ylabel
                    else "VL × Métrica de Validação"
                )
                ax.set_title(title)
                fig.tight_layout()
                return fig

            curves_plot = create_plot(build_curves)
            curves_title = "Curvas de validação cruzada"

        vip_rows: list[dict[str, str]] = []
        vip_list: list[dict] = []
        if isinstance(vip_data, list):
            vip_list = [
                entry
                for entry in vip_data
                if isinstance(entry, dict) and entry.get("score") is not None
            ]
        elif isinstance(vip_data, dict):
            scores = vip_data.get("scores")
            wavelengths = vip_data.get("wavelengths") or []
            if isinstance(scores, list):
                vip_list = [
                    {"wavelength": wavelengths[i] if i < len(wavelengths) else i, "score": score}
                    for i, score in enumerate(scores)
                ]
        vip_list.sort(key=lambda item: safe_float(item.get("score")) or 0, reverse=True)
        for entry in vip_list[:15]:
            vip_rows.append(
                {
                    "wavelength": format_number(entry.get("wavelength"), 1),
                    "score": format_number(entry.get("score"), 3),
                }
            )

        vip_plot = ""
        if vip_list:
            def build_vip(plt):
                top_entries = vip_list[:20]
                wavelengths = [format_number(e.get("wavelength"), 1) for e in top_entries]
                scores_vals = [safe_float(e.get("score")) or 0 for e in top_entries]
                fig, ax = plt.subplots(figsize=(6, 4))
                positions = list(range(len(top_entries)))[::-1]
                ax.barh(positions, scores_vals[::-1], color="#2563eb")
                ax.set_yticks(positions)
                ax.set_yticklabels(wavelengths[::-1])
                ax.set_xlabel("Score VIP")
                ax.set_ylabel("Comprimento de onda (nm)")
                ax.set_title("Top 20 VIP")
                ax.grid(True, axis="x", linestyle="--", alpha=0.3)
                fig.tight_layout()
                return fig

            vip_plot = create_plot(build_vip)

        confusion_plot = ""
        confusion_labels: list[str] = []
        confusion_rows: list[list[str]] = []
        if isinstance(confusion_matrix, dict):
            labels = confusion_matrix.get("labels") or []
            matrix = confusion_matrix.get("matrix") or []
            if labels and matrix:
                confusion_labels = [str(label) for label in labels]
                for row in matrix:
                    confusion_rows.append(
                        [format_number(value, 3) or "0" for value in row]
                    )

                def build_confusion(plt):
                    fig, ax = plt.subplots(figsize=(5, 4))
                    numeric = [
                        [safe_float(value) or 0 for value in row] for row in matrix
                    ]
                    im = ax.imshow(numeric, cmap="Blues")
                    ax.set_xticks(range(len(confusion_labels)))
                    ax.set_yticks(range(len(confusion_labels)))
                    ax.set_xticklabels(confusion_labels, rotation=45, ha="right")
                    ax.set_yticklabels(confusion_labels)
                    for i, row_values in enumerate(numeric):
                        for j, value in enumerate(row_values):
                            ax.text(
                                j,
                                i,
                                format_number(value, 2) or "0",
                                ha="center",
                                va="center",
                                color="#0f172a",
                            )
                    ax.set_xlabel("Previsto")
                    ax.set_ylabel("Real")
                    ax.set_title("Matriz de confusão")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    return fig

                confusion_plot = create_plot(build_confusion)

        diagnostic_summary: list[dict[str, str]] = []
        residual_plot = ""
        std_residual_plot = ""
        leverage_plot = ""
        distribution_plot = ""

        if isinstance(residuals, dict):
            residual_outliers = residuals.get("outliers") or []
            std_threshold = residuals.get("std_threshold") or residuals.get("threshold")
            diagnostic_summary.append(
                {
                    "label": "Resíduos extremos",
                    "value": format_value(len(residual_outliers)),
                    "description": "|resíduo padronizado| > "
                    + (format_number(std_threshold, 1) or "3"),
                }
            )

            preds = residuals.get("predicted") or residuals.get("y_pred")
            raws = residuals.get("raw") or residuals.get("residuals")
            sample_index = residuals.get("sample_index") or []
            std_values = residuals.get("standardized")
            outlier_set = {str(v) for v in residual_outliers}

            if preds and raws:
                def build_residual(plt):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    xs = []
                    ys = []
                    colors = []
                    for idx, (pred, res) in enumerate(zip(preds, raws)):
                        xs.append(safe_float(pred) or 0)
                        ys.append(safe_float(res) or 0)
                        sid = (
                            str(sample_index[idx])
                            if idx < len(sample_index)
                            else str(idx)
                        )
                        colors.append("#dc2626" if sid in outlier_set else "#2563eb")
                    ax.scatter(xs, ys, c=colors, s=25, alpha=0.85)
                    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
                    ax.set_xlabel(
                        "Probabilidade prevista"
                        if resolved_task == "classification"
                        else "Valor previsto"
                    )
                    ax.set_ylabel("Resíduo")
                    ax.set_title("Resíduos × previsto")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    fig.tight_layout()
                    return fig

                residual_plot = create_plot(build_residual)

            if std_values:
                def build_std(plt):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    xs = list(range(1, len(std_values) + 1))
                    ys = [safe_float(v) or 0 for v in std_values]
                    ax.scatter(xs, ys, color="#16a34a", s=22)
                    threshold_val = safe_float(std_threshold) or 3
                    ax.axhline(threshold_val, color="#dc2626", linestyle="--")
                    ax.axhline(-threshold_val, color="#dc2626", linestyle="--")
                    ax.set_xlabel("Índice da amostra")
                    ax.set_ylabel("Resíduo padronizado")
                    ax.set_title("Resíduos padronizados")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    fig.tight_layout()
                    return fig

                std_residual_plot = create_plot(build_std)

        if isinstance(influence, dict):
            high_leverage = influence.get("high_leverage") or []
            leverage_threshold = influence.get("leverage_threshold")
            diagnostic_summary.append(
                {
                    "label": "Leverage elevado",
                    "value": format_value(len(high_leverage)),
                    "description": "Limite: "
                    + (format_number(leverage_threshold, 3) or "-"),
                }
            )
            hotelling = influence.get("hotelling_outliers") or []
            hotelling_threshold = influence.get("hotelling_t2_threshold")
            if hotelling:
                diagnostic_summary.append(
                    {
                        "label": "Hotelling T²",
                        "value": format_value(len(hotelling)),
                        "description": "Acima de "
                        + (format_number(hotelling_threshold, 2) or "limite"),
                    }
                )

            leverage_values = influence.get("leverage")
            sample_index = influence.get("sample_index") or []
            if leverage_values:
                def build_leverage(plt):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    xs = (
                        [safe_float(idx) or i for i, idx in enumerate(sample_index, start=1)]
                        if sample_index
                        else list(range(1, len(leverage_values) + 1))
                    )
                    ys = [safe_float(v) or 0 for v in leverage_values]
                    ax.scatter(xs, ys, color="#0ea5e9", s=22)
                    threshold_val = safe_float(leverage_threshold)
                    if threshold_val is not None:
                        ax.axhline(threshold_val, color="#f97316", linestyle="--")
                    ax.set_xlabel("Índice da amostra")
                    ax.set_ylabel("Leverage")
                    ax.set_title("Diagnóstico de leverage")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    fig.tight_layout()
                    return fig

                leverage_plot = create_plot(build_leverage)

        if isinstance(distributions, dict):
            probabilities = distributions.get("probabilities")
            predicted_hist = distributions.get("predicted")

            if resolved_task == "classification" and isinstance(probabilities, dict):
                def build_prob(plt):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    bottoms = None
                    for label, hist in probabilities.items():
                        edges = hist.get("bin_edges") or []
                        counts = hist.get("counts") or []
                        if len(edges) < 2 or not counts:
                            continue
                        centers = [
                            (edges[i] + edges[i + 1]) / 2
                            for i in range(len(edges) - 1)
                        ]
                        width = (
                            edges[1] - edges[0] if len(edges) > 1 else 0.05
                        )
                        if bottoms is None:
                            bottoms = [0] * len(centers)
                        ax.bar(
                            centers,
                            counts,
                            width=width * 0.9,
                            bottom=bottoms,
                            alpha=0.7,
                            label=str(label),
                        )
                        bottoms = [b + c for b, c in zip(bottoms, counts)]
                    ax.set_xlabel("Probabilidade prevista")
                    ax.set_ylabel("Frequência")
                    ax.set_title("Distribuição das probabilidades")
                    ax.set_xlim(0, 1)
                    ax.legend(loc="best")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    fig.tight_layout()
                    return fig

                distribution_plot = create_plot(build_prob)

            elif predicted_hist:
                def build_pred(plt):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    edges = predicted_hist.get("bin_edges") or []
                    counts = predicted_hist.get("counts") or []
                    if len(edges) < 2 or not counts:
                        return None
                    centers = [
                        (edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)
                    ]
                    width = (
                        edges[1] - edges[0] if len(edges) > 1 else 0.1
                    )
                    ax.bar(centers, counts, width=width * 0.9, color="#2563eb")
                    ax.set_xlabel("Valor previsto")
                    ax.set_ylabel("Frequência")
                    ax.set_title("Distribuição das previsões")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    fig.tight_layout()
                    return fig

                distribution_plot = create_plot(build_pred)

        latent_plot = ""
        latent_r2_rows: list[dict[str, str]] = []
        if isinstance(latent, dict):
            scores = latent.get("scores") or []
            labels = latent.get("sample_labels") or []
            if scores:
                def build_latent(plt):
                    valid_scores = [row for row in scores if len(row) >= 2]
                    if not valid_scores:
                        return None
                    fig, ax = plt.subplots(figsize=(6, 4))
                    unique_labels = list(dict.fromkeys(labels)) if labels else []
                    if not unique_labels:
                        xs = [safe_float(row[0]) or 0 for row in valid_scores]
                        ys = [safe_float(row[1]) or 0 for row in valid_scores]
                        ax.scatter(xs, ys, color="#2563eb", s=22)
                    else:
                        cmap = plt.cm.get_cmap("tab10", len(unique_labels) + 1)
                        for idx, label in enumerate(unique_labels):
                            xs = []
                            ys = []
                            for pos, row in enumerate(valid_scores):
                                sample_label = labels[pos] if pos < len(labels) else ""
                                if sample_label == label:
                                    xs.append(safe_float(row[0]) or 0)
                                    ys.append(safe_float(row[1]) or 0)
                            color = cmap(idx)
                            ax.scatter(xs, ys, color=color, s=22, label=str(label))
                        ax.legend(loc="best")
                    ax.set_xlabel("LV1")
                    ax.set_ylabel("LV2")
                    ax.set_title("Scores latentes (LV1 × LV2)")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    fig.tight_layout()
                    return fig

                latent_plot = create_plot(build_latent)

            r2x = latent.get("r2x_cum") or []
            r2y = latent.get("r2y_cum") or []
            for idx, value in enumerate(r2x):
                latent_r2_rows.append(
                    {
                        "component": str(idx + 1),
                        "r2x": format_number(value, 4),
                        "r2y": format_number(r2y[idx] if idx < len(r2y) else None, 4),
                    }
                )

        goal_data = result.get("goal") or user_inputs.get("goal")
        goal_warning = result.get("goal_warning") or user_inputs.get("goal_warning")

        def goal_to_text(goal):
            if isinstance(goal, dict):
                metric_name = goal.get("metric") or goal.get("name") or "Métrica"
                comparison = goal.get("comparison") or goal.get("operator") or ">="
                target = goal.get("target") or goal.get("value")
                formatted_target = format_number(target, 3) if target is not None else ""
                if formatted_target:
                    return f"{metric_name} {comparison} {formatted_target}"
                return f"{metric_name} {comparison}"
            if goal:
                return str(goal)
            return ""

        goal_text = goal_to_text(goal_data)

        context = {
            "title": self.title,
            "metrics": metrics,
            "params": params,
            "logo": settings.assets_logo,
            "scatter_path": scatter_valid,
            "vip_path": vip_valid,
            "user": user or params.get("user", ""),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary_rows": summary_rows + validation_rows,
            "validation_used": result.get("validation_used"),
            "n_splits_effective": result.get("n_splits_effective"),
            "preprocess_list": preprocess_list,
            "preprocess_grid_rows": preprocess_grid_rows,
            "sg_description": sg_description,
            "train_metrics_rows": train_metrics_rows,
            "validation_metrics_rows": validation_metrics_rows,
            "metrics_rows": metrics_rows,
            "best": best_copy,
            "best_metrics_rows": best_metrics_rows,
            "per_class_rows": per_class_rows,
            "curves_plot": curves_plot,
            "curves_title": curves_title,
            "vip_plot": vip_plot,
            "vip_rows": vip_rows,
            "conf_path": conf_valid,
            "class_report_path": class_valid,
            "confusion_plot": confusion_plot,
            "confusion_labels": confusion_labels,
            "confusion_rows": confusion_rows,
            "diagnostic_summary": diagnostic_summary,
            "residual_plot": residual_plot,
            "std_residual_plot": std_residual_plot,
            "leverage_plot": leverage_plot,
            "distribution_plot": distribution_plot,
            "latent_plot": latent_plot,
            "latent_r2_rows": latent_r2_rows,
            "interpretacao_vips": interpretacao_vips,
            "resumo_interpretativo": resumo_interpretativo,
            "opt_results": opt_results or [],
            "opt_plot": (
                opt_plot_path if opt_plot_path and os.path.exists(opt_plot_path) else ""
            ),
            "range_used_display": format_range(range_source),
            "goal_text": goal_text,
            "goal_warning": goal_warning,
            "task": resolved_task,
        }

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
