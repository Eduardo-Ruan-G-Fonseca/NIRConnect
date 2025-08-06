# NIR Modernized v4.6 - Pacote Consolidado

Este pacote contém todos os módulos:
- PLSR e PLS-DA
- Otimização (iPLS)
- Bootstrapping
- Relatórios PDF e HTML
- APIs FastAPI e Flask (Locaweb)
- CLI


## Instalação
Execute `pip install -r requirements.txt` para instalar as dependências, incluindo o pacote `requests` utilizado nos testes.

## Testes
Execute `pytest tests/` para rodar os testes.


## Relatórios HTML -> PDF
Use `html_to_pdf.sh arquivo.html saida.pdf` ou o comando `python cli.py html-to-pdf arquivo.html saida.pdf`.
O comando `report` agora aceita `--engine wkhtml` para gerar PDFs com o `wkhtmltopdf` quando disponível.


## Monitoramento
Use `./monitor.sh` para checar o status da API local.


## Dashboard de Logs
Rode `python generate_dashboard.py` e abra `dashboard.html`.


## Gráficos de Logs
Agora o dashboard inclui gráficos interativos com Chart.js.


## Painel de Estatísticas
Dashboard agora inclui métricas de modelo (R², RMSE, Accuracy).

## Upload de Métricas
Envie métricas reais para o dashboard com:

```bash
curl -X POST http://127.0.0.1:5000/metrics/upload \
     -H "Content-Type: application/json" \
     -d '{"R2": 0.97, "RMSE": 0.08, "Accuracy": 0.92}'
```

Obtenha as métricas atuais com:

```bash
curl http://127.0.0.1:5000/metrics
```


## Dashboard Dinâmico
O dashboard agora atualiza logs e métricas automaticamente a cada 5 segundos.


## Filtros e Histórico
Agora é possível filtrar logs por tipo e data, além de visualizar histórico de métricas.

## Deploy na Locaweb
Para rodar a API Flask em um servidor WSGI (como na Locaweb), utilize:

```bash
sh run_wsgi.sh
```
O script ativa o ambiente virtual caso exista e executa o Gunicorn com o aplicativo definido em `public_html/api/main.py`.
## Servidor Local
Inicie rapidamente a API FastAPI com:

```bash
python cli.py serve --host 0.0.0.0 --port 8000
```

Gere o dashboard e abra o HTML resultante:

```bash
python cli.py dashboard dashboard.html
```

Para processar um arquivo Excel diretamente pela API, envie uma requisição POST
para `/process`:

```bash
curl -F "file=@dados.xlsx" -F "target=coluna_alvo" \
     -F "n_components=5" -F "classification=false" \
     http://localhost:8000/process
```

Crie relatórios em PDF a partir de métricas JSON:

```bash
python cli.py report '{"R2": 0.95, "RMSE": 0.08, "Accuracy": 0.9}' --output relatorio.pdf
```

## Inspecionar Arquivos
Utilize o endpoint `/columns` para identificar quais colunas do arquivo são
espectrais e quais são potenciais alvos, além de obter a média e matriz de
espectros.

```bash
curl -F "file=@dados.csv" http://localhost:8000/columns
```

## Análise com Validação
Para executar PLS-R ou PLS-DA com validação cruzada, envie o arquivo para
`/analisar` indicando o alvo, número de componentes e eventuais etapas de
pré-processamento:

```bash
curl -F "file=@dados.csv" -F "target=coluna_alvo" \
     -F "n_components=5" -F "preprocess=zscore" \
     -F "validation_method=KFold" \
     http://localhost:8000/analisar
```

É possível encadear métodos de pré-processamento e definir parâmetros das
derivadas passando JSON no campo `preprocess`, por exemplo
`{"method": "sg1", "params": {"window_length": 15, "polyorder": 2}}`.

## Otimização Automática
O endpoint `/optimize` realiza buscas sobre combinações de métodos de
pré-processamento e número de componentes. Consulte `/optimize/status` para
acompanhar o progresso da rotina.

## Treinamento via CLI
Além da API, a ferramenta oferece um comando para treinar modelos
diretamente no terminal:

```bash
python cli.py train dados.csv target --n-components 5
```
## Licença
Este projeto utiliza a licença MIT. Consulte o arquivo LICENSE para mais informações.
