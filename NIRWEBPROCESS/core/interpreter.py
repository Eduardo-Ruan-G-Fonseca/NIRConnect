from core.tabela_bandas import BANDAS_NIR


def interpretar_vips(vips: list, wavelengths: list, top_n: int = 10) -> list:
    """Retorna interpretação dos top_n VIPs mais relevantes."""
    top_indices = sorted(range(len(vips)), key=lambda i: -vips[i])[:top_n]
    resultados = []

    for i in top_indices:
        wl = wavelengths[i]
        vip_val = round(vips[i], 3)
        faixa_quimica = None
        if isinstance(wl, (int, float)) and not (isinstance(wl, float) and wl != wl):
            faixa_quimica = next(
                (
                    banda
                    for banda in BANDAS_NIR
                    if banda["wavelength_range"][0] <= wl <= banda["wavelength_range"][1]
                ),
                None,
            )

        resultados.append(
            {
                "comprimento_onda_nm": wl,
                "vip": vip_val,
                "grupo": faixa_quimica["grupo"] if faixa_quimica else "Desconhecido",
                "vibracao": faixa_quimica["vibracao"] if faixa_quimica else "-",
                "comentario": faixa_quimica["comentario"] if faixa_quimica else "Sem correspondência química direta",
            }
        )

    return resultados


def gerar_resumo_interpretativo(interpretacoes: list[dict]) -> str:
    """Gera texto sumário a partir da lista de interpretações de VIPs."""
    total = len(interpretacoes)
    contagem: dict[str, int] = {}
    for item in interpretacoes:
        grupo = item.get("grupo")
        if grupo and grupo.lower() != "desconhecido":
            contagem[grupo] = contagem.get(grupo, 0) + 1

    if not contagem:
        return (
            "As regiões espectrais mais relevantes não correspondem diretamente a "
            "grupos químicos conhecidos na base de dados."
        )

    ordenados = sorted(contagem.items(), key=lambda x: -x[1])
    partes = []
    for idx, (grupo, qtd) in enumerate(ordenados):
        if idx == 0:
            partes.append(
                f"{grupo} (presente em {qtd} das {total} faixas mais influentes)"
            )
        else:
            partes.append(f"{grupo} ({qtd})")

    return (
        "O modelo mostrou maior sensibilidade às seguintes classes químicas: "
        + ", ".join(partes)
        + "."
    )
