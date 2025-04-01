# 📊 Previsão de Preços com LSTM + Transformer + GRU

Este projeto utiliza aprendizado profundo com redes neurais híbridas (LSTM, Transformer e GRU) para realizar previsões de preços de ações com base em dados do Yahoo Finance. Ele também gera gráficos detalhados de análise técnica e estratégias de trading como Swing Trade, Scalping e Holding.

---

## 🔍 Funcionalidades

- Coleta de dados financeiros históricos via Yahoo Finance
- Engenharia de atributos com indicadores técnicos (RSI, MACD, Bollinger Bands, etc.)
- Criação de modelo preditivo híbrido (LSTM + Transformer + GRU)
- Previsões futuras de 90 dias
- Estratégias automatizadas:
  - Swing Trade
  - Scalping
  - Posição de Longo Prazo (Holding)
- Geração de gráficos técnicos e relatórios automáticos

---

## 🧠 Tecnologias Utilizadas

- Python 3.8+
- [TensorFlow](https://www.tensorflow.org/) e [Keras](https://keras.io/)
- [yfinance](https://pypi.org/project/yfinance/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [ta](https://pypi.org/project/ta/) (technical analysis)
- [mplfinance](https://github.com/matplotlib/mplfinance)

---
## ⚙️ Instalação

 Instale as dependências:

```bash
pip install -r requirements.txt
```

**Se não tiver um `requirements.txt`, instale manualmente:**

```bash
pip install pandas numpy yfinance matplotlib mplfinance scikit-learn tensorflow ta
```

---

## 🚀 Como Usar

1. Edite o código para inserir o ticker da ação desejada (ex: `VALE3.SA`, `PETR4.SA`, `AAPL`, etc.)
2. Execute o script `modelo_atualizado.py`
3. O script irá:
   - Baixar os dados
   - Criar e treinar o modelo
   - Fazer previsões
   - Gerar gráficos e relatórios automáticos

---

## 📈 Exemplos de Saída

- `retorno_mensal_barras_<ticker>.png`
- `swing_trade_candlestick_<ticker>.png`
- `holding_position_candlestick_<ticker>.png`
- `estrategia_detalhada_<ticker>.png`
- `relatorio_recomendacoes_<ticker>.png`

