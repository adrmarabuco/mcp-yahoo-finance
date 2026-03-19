# Yahoo Finance MCP Server

Um servidor MCP (Model Context Protocol) completo para acessar dados do Yahoo Finance, permitindo que LLMs como o Claude interajam com dados de mercado financeiro em tempo real.

## 🚀 Funcionalidades

| Ferramenta | Descrição |
|------------|-----------|
| `yf_get_quote` | Cotação em tempo real e informações básicas |
| `yf_get_multiple_quotes` | Cotações de múltiplas ações |
| `yf_get_historical_data` | Dados históricos OHLCV (Open, High, Low, Close, Volume) |
| `yf_get_financials` | Demonstrativos financeiros (DRE, Balanço, Fluxo de Caixa) |
| `yf_get_options` | Cadeia de opções (calls e puts) |
| `yf_get_dividends` | Histórico e informações de dividendos |
| `yf_get_recommendations` | Recomendações de analistas e preços-alvo |
| `yf_get_news` | Notícias recentes relacionadas |
| `yf_get_holders` | Principais acionistas institucionais |
| `yf_get_earnings` | Histórico de lucros e calendário de earnings |
| `yf_get_calendar` | Calendário de earnings e próximas datas |
| `yf_compare_stocks` | Comparação lado a lado de múltiplas ações |

## 📦 Instalação

### Via pip

```bash
pip install mcp yfinance httpx pydantic
```

### Via requirements.txt

```bash
pip install -r requirements.txt
```

### Via pyproject.toml (desenvolvimento)

```bash
pip install -e ".[dev]"
```

## ⚙️ Configuração

### Claude Desktop

Adicione ao seu arquivo de configuração do Claude Desktop (`claude_desktop_config.json`):

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "yahoo-finance": {
      "command": "python",
      "args": ["/caminho/para/yahoo_finance_mcp.py"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add yahoo-finance python /caminho/para/yahoo_finance_mcp.py
```

### Uso direto via stdio

```bash
python yahoo_finance_mcp.py
```

### Uso via HTTP (servidor remoto)

Modifique a última linha do arquivo para:

```python
if __name__ == "__main__":
    mcp.run(transport="streamable_http", port=8000)
```

## 📖 Exemplos de Uso

### Obter cotação de uma ação

```
Qual é a cotação atual da AAPL?
```

A ferramenta `yf_get_quote` retornará:
- Preço atual priorizando `postMarketPrice`/`preMarketPrice` quando disponíveis
- Campos explícitos para `regular_market_price`, `pre_market_price` e `post_market_price`
- Origem do preço em `current_price_source` para distinguir regular, pre-market e after-hours
- Volume e volume médio
- Market cap
- P/E ratio e outras métricas
- Setor e indústria

### Dados históricos para backtesting

```
Me dê os dados históricos do BTC-USD dos últimos 3 meses com intervalo diário
```

A ferramenta `yf_get_historical_data` retornará OHLCV para o período solicitado.

### Análise de opções

```
Mostre a cadeia de opções da TSLA
```

A ferramenta `yf_get_options` retornará calls e puts com:
- Strike prices
- Prêmios (bid/ask)
- Volume e open interest
- Volatilidade implícita

### Comparação de ações

```
Compare AAPL, MSFT e GOOGL
```

A ferramenta `yf_compare_stocks` criará uma tabela comparativa com métricas fundamentalistas.

## 🔧 Parâmetros das Ferramentas

### Períodos disponíveis (historical_data)
- `1d`, `5d` - Dias
- `1mo`, `3mo`, `6mo` - Meses
- `1y`, `2y`, `5y`, `10y` - Anos
- `ytd` - Year to date
- `max` - Todo histórico disponível

### Intervalos disponíveis (historical_data)
- `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h` - Intraday
- `1d`, `5d`, `1wk`, `1mo`, `3mo` - Períodos maiores

### Formatos de resposta
- `json` - Para processamento programático
- `markdown` - Para leitura humana

## 🛡️ Limitações

- **Rate Limits:** O Yahoo Finance pode aplicar limites de requisições
- **Dados Atrasados:** Cotações podem ter delay de 15-20 minutos
- **Disponibilidade:** Alguns dados podem não estar disponíveis para todas as ações
- **Termos de Uso:** Respeite os termos de uso do Yahoo Finance

## 🧪 Testando

```bash
# Verificar sintaxe
python -m py_compile yahoo_finance_mcp.py

# Executar o servidor
python yahoo_finance_mcp.py --help

# Testar com MCP Inspector
npx @modelcontextprotocol/inspector python yahoo_finance_mcp.py
```

## 📁 Estrutura do Projeto

```
yahoo_finance_mcp/
├── yahoo_finance_mcp.py   # Servidor MCP principal
├── requirements.txt       # Dependências
├── pyproject.toml        # Configuração do pacote
└── README.md             # Esta documentação
```

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

MIT License - veja o arquivo LICENSE para detalhes.

## 🔗 Links Úteis

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [yfinance Documentation](https://ranaroussi.github.io/yfinance/)
- [FastMCP Documentation](https://github.com/modelcontextprotocol/python-sdk)
