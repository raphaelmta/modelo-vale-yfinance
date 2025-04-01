#!/usr/bin/env python
# coding: utf-8

# Tratamento de erros para importações
try:
    import os
    import sys
    import time
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf
    import datetime
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    import ta
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    import mplfinance as mpf  # Adicionado para gráficos de candlestick

    # Configurações para caracteres em português
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.family': 'DejaVu Sans'})

    # Configuração do matplotlib para modo não-interativo
    plt.switch_backend('Agg')

    # Suprime avisos
    warnings.filterwarnings("ignore")

    print("Todas as bibliotecas importadas com sucesso!")
    print(f"Versão do TensorFlow: {tf.__version__}")
    if tf.__version__.startswith('2'):
        print("Versão compatível do TensorFlow detectada.")
    else:
        print("Aviso: Este código foi testado com TensorFlow 2.x. Sua versão pode não ser compatível.")
    
except ImportError as e:
    print(f"Erro ao importar biblioteca: {e}")
    print("Execute 'pip install ta scikit-learn pandas numpy tensorflow yfinance matplotlib mplfinance' para instalar as dependências")
    sys.exit(1)

# Função corrigida para extrair dados com tratamento de erros para MultiIndex
def rmta_extrai_dados(ticker):
    try:
        print(f"Baixando dados para {ticker} via Yahoo Finance...")
        # Definir auto_adjust=True explicitamente para evitar avisos
        dados = yf.download(ticker, start="2010-01-01", end=datetime.datetime.now().strftime("%Y-%m-%d"), auto_adjust=True)
        
        print("Número de colunas retornadas:", len(dados.columns))
        print("Nomes originais das colunas:", dados.columns)

        # Verificar se temos um MultiIndex
        if isinstance(dados.columns, pd.MultiIndex):
            # Extrair apenas o primeiro nível (Price) e converter para minúsculas
            dados.columns = [col[0].lower() for col in dados.columns]
        else:
            # Caso contrário, apenas converter para minúsculas
            dados.columns = [col.lower() for col in dados.columns]
        
        print(f"Colunas após processamento: {dados.columns}")
        
        # Verificar e renomear colunas conforme necessário
        colunas_esperadas = {"open", "high", "low", "close", "volume"}
        colunas_atuais = set(dados.columns)
        
        # Verificar se todas as colunas esperadas estão presentes
        if not colunas_esperadas.issubset(colunas_atuais):
            print(f"Aviso: Algumas colunas esperadas não estão presentes. Colunas atuais: {colunas_atuais}")
        
        # Renomear 'adj close' para 'adj_close' se existir
        if 'adj close' in dados.columns:
            dados = dados.rename(columns={'adj close': 'adj_close'})
        
        dados.index.name = "date"
        
        print(f"Dados baixados com sucesso. Preço de fechamento mais recente: $ {dados['close'].iloc[-1]:.2f}")
        
        # Extrair nome da empresa e informações adicionais
        try:
            info = yf.Ticker(ticker).info
            nome_empresa = info.get('shortName', ticker)
            setor = info.get('sector', 'N/A')
            industria = info.get('industry', 'N/A')
            print(f"Empresa: {nome_empresa}")
            print(f"Setor: {setor}")
            print(f"Indústria: {industria}")
        except:
            nome_empresa = f"Vale S.A. ({ticker})"
            setor = "Materiais Básicos"
            industria = "Mineração"
        
        return dados, nome_empresa, setor, industria
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        print("Verifique sua conexão com a internet ou tente outro ticker.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Função para engenharia de atributos aprimorada
def rmta_func_engenharia_atributos(df):
    try:
        print("Iniciando engenharia de atributos...")
        df_copy = df.copy()
        df_copy["retorno"] = df_copy["close"].pct_change(1)
        df_copy["op"] = df_copy["open"].shift(1)
        df_copy["hi"] = df_copy["high"].shift(1)
        df_copy["lo"] = df_copy["low"].shift(1)
        df_copy["clo"] = df_copy["close"].shift(1)
        df_copy["vol"] = df_copy["volume"].shift(1)
        
        # Médias móveis
        df_copy["SMA 15"] = df_copy[["close"]].rolling(15).mean().shift(1)
        df_copy["SMA 60"] = df_copy[["close"]].rolling(60).mean().shift(1)
        df_copy["SMA 200"] = df_copy[["close"]].rolling(200).mean().shift(1)  # Adicionado SMA de 200 dias
        df_copy["MSD 15"] = df_copy["retorno"].rolling(15).std().shift(1)
        df_copy["MSD 60"] = df_copy["retorno"].rolling(60).std().shift(1)

        # Indicadores adicionais
        vwap = ta.volume.VolumeWeightedAveragePrice(high=df['high'],
                                                   low=df['low'],
                                                   close=df['close'],
                                                   volume=df['volume'],
                                                   window=5)
        df_copy["VWAP"] = vwap.vwap.shift(1)

        # RSI - Relative Strength Index
        RSI = ta.momentum.RSIIndicator(df_copy["close"], window=14, fillna=False)
        df_copy["RSI"] = RSI.rsi().shift(1)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high=df_copy["high"], 
                                                low=df_copy["low"], 
                                                close=df_copy["close"], 
                                                window=14, 
                                                smooth_window=3)
        df_copy["Stoch_K"] = stoch.stoch().shift(1)
        df_copy["Stoch_D"] = stoch.stoch_signal().shift(1)
        
        # MACD - Moving Average Convergence Divergence
        macd = ta.trend.MACD(close=df_copy["close"], window_slow=26, window_fast=12, window_sign=9)
        df_copy["MACD"] = macd.macd().shift(1)
        df_copy["MACD_signal"] = macd.macd_signal().shift(1)
        df_copy["MACD_diff"] = macd.macd_diff().shift(1)
        
        # Bandas de Bollinger
        bollinger = ta.volatility.BollingerBands(close=df_copy["close"], window=20, window_dev=2)
        df_copy["BB_high"] = bollinger.bollinger_hband().shift(1)
        df_copy["BB_low"] = bollinger.bollinger_lband().shift(1)
        df_copy["BB_width"] = ((df_copy["BB_high"] - df_copy["BB_low"]) / df_copy["SMA 15"]).shift(1)
        
        # ATR - Average True Range (indicador de volatilidade)
        atr = ta.volatility.AverageTrueRange(high=df_copy["high"], 
                                            low=df_copy["low"], 
                                            close=df_copy["close"], 
                                            window=14)
        df_copy["ATR"] = atr.average_true_range().shift(1)
        
        # OBV - On-Balance Volume (indicador de volume)
        obv = ta.volume.OnBalanceVolumeIndicator(close=df_copy["close"], 
                                                volume=df_copy["volume"])
        df_copy["OBV"] = obv.on_balance_volume().shift(1)
        
        # Ichimoku Cloud (componentes principais)
        ichimoku = ta.trend.IchimokuIndicator(high=df_copy["high"], 
                                             low=df_copy["low"], 
                                             window1=9, 
                                             window2=26, 
                                             window3=52)
        df_copy["Ichimoku_A"] = ichimoku.ichimoku_a().shift(1)
        df_copy["Ichimoku_B"] = ichimoku.ichimoku_b().shift(1)
        
        # Características de preço
        df_copy["Price_Change"] = df_copy["close"].pct_change(5).shift(1)  # Mudança percentual de 5 dias
        df_copy["Price_Momentum"] = df_copy["close"].diff(5).shift(1)  # Momentum de 5 dias
        
        # Características de volume
        df_copy["Volume_Change"] = df_copy["volume"].pct_change(5).shift(1)  # Mudança percentual de volume de 5 dias
        df_copy["Volume_MA_Ratio"] = df_copy["volume"] / df_copy["volume"].rolling(20).mean().shift(1)  # Volume relativo à média
        
        # Características de volatilidade
        df_copy["Volatility"] = df_copy["retorno"].rolling(30).std().shift(1)  # Volatilidade de 30 dias
        
        # Características de tendência
        df_copy["Trend_Strength"] = abs(df_copy["SMA 15"] - df_copy["SMA 60"]) / df_copy["SMA 15"]
        df_copy["Trend_Direction"] = np.sign(df_copy["SMA 15"] - df_copy["SMA 60"])
        
        # Características sazonais
        df_copy["Day_of_Week"] = df_copy.index.dayofweek
        df_copy["Month"] = df_copy.index.month

        print("Engenharia de atributos concluída com sucesso.")
        return df_copy.dropna()
    except Exception as e:
        print(f"Erro na engenharia de atributos: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Função para ajustar o formato dos dados
def rmta_ajusta_formato_dados(X_s, y_s, lag):
    try:
        print(f"Ajustando formato dos dados com lag={lag}...")
        if len(X_s) != len(y_s):
            print("Aviso: X_s e y_s têm comprimentos diferentes")

        X_train = []
        
        for variable in range(0, X_s.shape[1]):
            X = []
            for i in range(lag, X_s.shape[0]):
                X.append(X_s[i-lag:i, variable])
            X_train.append(X)
        
        X_train = np.array(X_train)
        X_train = np.swapaxes(np.swapaxes(X_train, 0, 1), 1, 2)

        y_train = []
        for i in range(lag, y_s.shape[0]):
            y_train.append(y_s[i, :].reshape(-1,1).transpose())
        
        y_train = np.concatenate(y_train, axis=0)
        
        print("Formato dos dados ajustado com sucesso.")
        return X_train, y_train
    except Exception as e:
        print(f"Erro ao ajustar formato dos dados: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Função do transformer encoder aprimorada
def rmta_transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    try:
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
    except Exception as e:
        print(f"Erro no transformer encoder: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Função de criação do modelo aprimorada
def rmta_cria_modelo(input_shape, 
                   head_size, 
                   num_heads, 
                   ff_dim, 
                   num_transformer_blocks, 
                   mlp_units, 
                   dropout=0, 
                   mlp_dropout=0):
    try:
        print("Criando modelo aprimorado...")
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Camada LSTM bidirecional para capturar padrões temporais em ambas as direções
        x = layers.Bidirectional(layers.LSTM(20, return_sequences=True))(x)
        
        # Blocos Transformer para atenção
        for _ in range(num_transformer_blocks):
            x = rmta_transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        
        # Camada GRU para processamento sequencial
        x = layers.GRU(128, return_sequences=False)(x)
        x = layers.Dropout(mlp_dropout)(x)
        
        # Camadas densas para aprendizado de características
        x = layers.Dense(mlp_units, activation="relu")(x)
        x = layers.Dropout(mlp_dropout/2)(x)
        x = layers.Dense(mlp_units//2, activation="relu")(x)
        
        # Camada de saída
        outputs = layers.Dense(1)(x)
        
        print("Modelo aprimorado criado com sucesso.")
        return keras.Model(inputs, outputs)
    except Exception as e:
        print(f"Erro ao criar modelo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Nova função para gerar previsões futuras para 3 meses (90 dias)
def gerar_previsoes_futuras(modelo, df, features, sc, lag, dias_futuros=90):
    try:
        print(f"Gerando previsões para os próximos {dias_futuros} dias...")
        
        # Criar um DataFrame para armazenar as previsões futuras
        ultima_data = df.index[-1]
        datas_futuras = []
        
        # Gerar datas futuras, pulando finais de semana
        data_atual = ultima_data
        dias_adicionados = 0
        
        while dias_adicionados < dias_futuros:
            data_atual = data_atual + datetime.timedelta(days=1)
            # Pular finais de semana (5 = sábado, 6 = domingo)
            if data_atual.weekday() < 5:  
                datas_futuras.append(data_atual)
                dias_adicionados += 1
        
        df_futuro = pd.DataFrame(index=datas_futuras)
        
        # Inicializar com os últimos valores conhecidos
        df_temp = df.copy()
        
        # Fazer previsões dia a dia
        for i in range(dias_futuros):
            # Preparar os dados para previsão
            x_ultimo = df_temp[features].iloc[-lag:].values
            x_ultimo_sc = sc.transform(x_ultimo)
            
            # Ajustar formato para o modelo
            x_pred = np.array([x_ultimo_sc])
            x_pred = np.swapaxes(np.swapaxes(x_pred, 0, 1), 1, 2)
            
            # Fazer previsão
            pred = modelo.predict(x_pred, verbose=0)[0][0]
            
            # Criar uma nova linha com a previsão
            nova_linha = df_temp.iloc[-1:].copy()
            nova_linha.index = [datas_futuras[i]]
            nova_linha['retorno'] = pred
            
            # Calcular o novo preço com base no retorno previsto
            ultimo_preco = df_temp['close'].iloc[-1]
            novo_preco = ultimo_preco * (1 + pred)
            nova_linha['close'] = novo_preco
            
            # Atualizar outros valores (simplificado)
            nova_linha['open'] = novo_preco * 0.99  # Simplificação
            nova_linha['high'] = novo_preco * 1.01  # Simplificação
            nova_linha['low'] = novo_preco * 0.98   # Simplificação
            nova_linha['volume'] = df_temp['volume'].mean()  # Simplificação
            
            # Adicionar a nova linha ao DataFrame temporário
            df_temp = pd.concat([df_temp, nova_linha])
            
            # Recalcular indicadores técnicos
            df_temp = rmta_func_engenharia_atributos(df_temp)
        
        # Extrair apenas as linhas futuras
        df_futuro = df_temp.iloc[-dias_futuros:]
        
        # Calcular retornos acumulados para as previsões
        df_futuro['retorno_acumulado'] = df_futuro['retorno'].cumsum()
        
        # Adicionar sinais de negociação
        df_futuro['sinal'] = np.sign(df_futuro['retorno'])
        
        print("Previsões futuras geradas com sucesso.")
        return df_futuro
    except Exception as e:
        print(f"Erro ao gerar previsões futuras: {e}")
        import traceback
        traceback.print_exc()
        return None

# Função para criar gráfico de barras para retornos mensais
def criar_grafico_barras_retorno_mensal(df, split_val, ticker, nome_ativo):
    try:
        print("Criando gráfico de barras para retornos mensais...")
        
        # Criar DataFrame apenas com os dados de teste
        df_teste = df.iloc[split_val:].copy()
        
        # Calcular retornos mensais
        df_teste['ano_mes'] = df_teste.index.to_period('M')
        
        # Calcular retornos mensais para Buy & Hold e Estratégia
        retornos_mensais_bh = df_teste.groupby('ano_mes')['retorno'].sum()
        retornos_mensais_estrategia = df_teste.groupby('ano_mes')['estrategia'].sum()
        
        # Criar figura
        plt.figure(figsize=(15, 8))
        
        # Converter Period para datetime para plotagem
        datas = [pd.Period(p).to_timestamp() for p in retornos_mensais_bh.index]
        
        # Plotar barras
        bar_width = 10  # em dias
        plt.bar([d - datetime.timedelta(days=bar_width/2) for d in datas], 
                retornos_mensais_bh.values * 100, 
                width=bar_width, 
                label='Comprar e Manter', 
                color='blue', 
                alpha=0.7)
        
        plt.bar([d + datetime.timedelta(days=bar_width/2) for d in datas], 
                retornos_mensais_estrategia.values * 100, 
                width=bar_width, 
                label='Estratégia', 
                color='green', 
                alpha=0.7)
        
        # Adicionar linha de retorno acumulado
        ax2 = plt.twinx()
        retorno_acumulado_bh = df_teste['retorno'].cumsum() * 100
        retorno_acumulado_estrategia = df_teste['estrategia'].cumsum() * 100
        
        ax2.plot(df_teste.index, retorno_acumulado_bh, 
                color='darkblue', linestyle='--', linewidth=2, 
                label='Comprar e Manter Acumulado')
        
        ax2.plot(df_teste.index, retorno_acumulado_estrategia, 
                color='darkgreen', linestyle='--', linewidth=2, 
                label='Estratégia Acumulada')
        
        # Configurar eixos e legendas
        plt.title(f'Retornos Mensais e Acumulados - {nome_ativo}', fontsize=16)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Retorno Mensal (%)', fontsize=12)
        ax2.set_ylabel('Retorno Acumulado (%)', fontsize=12)
        
        # Formatar eixo x para mostrar apenas meses
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # Combinar legendas de ambos os eixos
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        # Adicionar grade
        plt.grid(True, alpha=0.3)
        
        # Adicionar explicação da estratégia
        plt.figtext(0.5, 0.01, 
                  "COMPARAÇÃO DE ESTRATÉGIAS DE NEGOCIAÇÃO\n\n"
                  "POSIÇÃO DE LONGO PRAZO: Estratégia de longo prazo que mantém a posição independente de flutuações de curto prazo.\n"
                  "SWING TRADE: Estratégia que utiliza aprendizado profundo para prever a direção do movimento de preço e opera em ambas direções.\n"
                  "O desempenho é medido comparando os retornos acumulados das estratégias com a abordagem passiva de Comprar e Manter.",
                  fontsize=10, ha='center', bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(f'retorno_mensal_barras_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo como 'retorno_mensal_barras_{ticker}.png'")
        
        return True
    except Exception as e:
        print(f"Erro ao criar gráfico de barras: {e}")
        import traceback
        traceback.print_exc()
        return False

# Nova função para criar gráfico de candlestick para a estratégia de swing trade
def criar_grafico_candlestick_swing_trade(df, df_futuro, split_val, ticker, nome_ativo):
    try:
        print("Criando gráfico de candlestick para estratégia de swing trade...")
        
        # Selecionar os últimos 120 dias para visualização
        dias_visualizacao = 120
        df_recente = df.iloc[-dias_visualizacao:].copy()
        
        # Preparar dados para o gráfico de candlestick
        df_ohlc = df_recente[['open', 'high', 'low', 'close']].copy()
        
        # Adicionar volume
        df_ohlc['volume'] = df_recente['volume']
        
        # Adicionar médias móveis
        df_ohlc['SMA15'] = df_recente['SMA 15']
        df_ohlc['SMA60'] = df_recente['SMA 60']
        
        # Criar sinais de compra e venda baseados na previsão
        sinais = np.sign(df_recente["prediction"].shift(1))
        compras = pd.Series(np.where(sinais > 0, df_recente['low'] * 0.99, np.nan), index=df_recente.index)
        vendas = pd.Series(np.where(sinais < 0, df_recente['high'] * 1.01, np.nan), index=df_recente.index)
        
        # Adicionar anotações para os sinais
        apds = [
            mpf.make_addplot(df_ohlc['SMA15'], color='blue', width=1, label='Média Móvel 15'),
            mpf.make_addplot(df_ohlc['SMA60'], color='red', width=1, label='Média Móvel 60'),
            mpf.make_addplot(compras, type='scatter', markersize=100, marker='^', color='green', label='Compra'),
            mpf.make_addplot(vendas, type='scatter', markersize=100, marker='v', color='red', label='Venda')
        ]
        
        # Configurar estilo do gráfico
        mc = mpf.make_marketcolors(
            up='green', down='red',
            edge='inherit',
            wick={'up':'green', 'down':'red'},
            volume='blue'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            figsize=(15, 10),
            gridstyle='--',
            y_on_right=False,
            volume_alpha=0.5
        )
        
        # Criar figura e salvar
        fig, axes = mpf.plot(
            df_ohlc,
            type='candle',
            style=s,
            addplot=apds,
            volume=True,
            panel_ratios=(4, 1),
            title=f'Estratégia de Swing Trade - {nome_ativo}',
            ylabel='Preço (R$)',
            ylabel_lower='Volume',
            returnfig=True
        )
        
        # Adicionar legenda
        axes[0].legend(loc='upper left')
        
        # Adicionar explicação da estratégia
        fig.text(0.5, 0.01, 
                "ESTRATÉGIA DE SWING TRADE\n\n"
                "Esta estratégia utiliza um modelo híbrido LSTM-Transformer-GRU para prever movimentos de curto prazo.\n"
                "Sinais de COMPRA (triângulos verdes) são gerados quando o modelo prevê um retorno positivo.\n"
                "Sinais de VENDA (triângulos vermelhos) são gerados quando o modelo prevê um retorno negativo.\n"
                "O objetivo é capturar movimentos de preço em ambas as direções em períodos de 1-5 dias.",
                fontsize=10, ha='center', bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(f'swing_trade_candlestick_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo como 'swing_trade_candlestick_{ticker}.png'")
        
        return True
    except Exception as e:
        print(f"Erro ao criar gráfico de candlestick para swing trade: {e}")
        import traceback
        traceback.print_exc()
        return False

# Nova função para criar gráfico de candlestick para a estratégia de holding position
def criar_grafico_candlestick_holding(df, df_futuro, split_val, ticker, nome_ativo):
    try:
        print("Criando gráfico de candlestick para estratégia de posição de longo prazo...")
        
        # Selecionar dados de longo prazo (últimos 2 anos)
        dias_visualizacao = 252 * 2  # Aproximadamente 2 anos de dias úteis
        if len(df) > dias_visualizacao:
            df_longo_prazo = df.iloc[-dias_visualizacao:].copy()
        else:
            df_longo_prazo = df.copy()
        
        # Preparar dados para o gráfico de candlestick
        df_ohlc = df_longo_prazo[['open', 'high', 'low', 'close']].copy()
        
        # Adicionar volume
        df_ohlc['volume'] = df_longo_prazo['volume']
        
        # Adicionar médias móveis de longo prazo
        df_ohlc['SMA60'] = df_longo_prazo['SMA 60']
        df_ohlc['SMA200'] = df_longo_prazo['SMA 200']
        
        # Identificar pontos de entrada
        # Identificar pontos de entrada para estratégia de holding (cruzamento de médias móveis)
        entradas = pd.Series(np.where((df_longo_prazo['SMA 60'].shift(1) < df_longo_prazo['SMA 200'].shift(1)) & 
                                     (df_longo_prazo['SMA 60'] > df_longo_prazo['SMA 200']), 
                                     df_longo_prazo['low'] * 0.99, np.nan), index=df_longo_prazo.index)
        
        # Identificar pontos de saída para estratégia de holding
        saidas = pd.Series(np.where((df_longo_prazo['SMA 60'].shift(1) > df_longo_prazo['SMA 200'].shift(1)) & 
                               (df_longo_prazo['SMA 60'] < df_longo_prazo['SMA 200']), 
                               df_longo_prazo['high'] * 1.01, np.nan), index=df_longo_prazo.index)
        
        # Adicionar anotações para os sinais
        apds = [
            mpf.make_addplot(df_ohlc['SMA60'], color='blue', width=1.5, label='Média Móvel 60'),
            mpf.make_addplot(df_ohlc['SMA200'], color='red', width=1.5, label='Média Móvel 200'),
            mpf.make_addplot(entradas, type='scatter', markersize=150, marker='^', color='green', label='Entrada'),
            mpf.make_addplot(saidas, type='scatter', markersize=150, marker='v', color='red', label='Saída')
        ]
        
        # Configurar estilo do gráfico
        mc = mpf.make_marketcolors(
            up='green', down='red',
            edge='inherit',
            wick={'up':'green', 'down':'red'},
            volume='blue'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            figsize=(15, 10),
            gridstyle='--',
            y_on_right=False,
            volume_alpha=0.5
        )
        
        # Criar figura e salvar
        fig, axes = mpf.plot(
            df_ohlc,
            type='candle',
            style=s,
            addplot=apds,
            volume=True,
            panel_ratios=(4, 1),
            title=f'Estratégia de Posição de Longo Prazo - {nome_ativo}',
            ylabel='Preço (R$)',
            ylabel_lower='Volume',
            returnfig=True
        )
        
        # Adicionar legenda
        axes[0].legend(loc='upper left')
        
        # Adicionar explicação da estratégia
        fig.text(0.5, 0.01, 
                "ESTRATÉGIA DE POSIÇÃO DE LONGO PRAZO\n\n"
                "Esta estratégia de longo prazo utiliza o cruzamento de médias móveis para identificar tendências duradouras.\n"
                "Sinais de ENTRADA (triângulos verdes) são gerados quando a MM 60 cruza para cima a MM 200 (Golden Cross).\n"
                "Sinais de SAÍDA (triângulos vermelhos) são gerados quando a MM 60 cruza para baixo a MM 200 (Death Cross).\n"
                "O objetivo é capturar tendências de longo prazo e manter a posição por meses ou anos.",
                fontsize=10, ha='center', bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(f'holding_position_candlestick_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo como 'holding_position_candlestick_{ticker}.png'")
        
        return True
    except Exception as e:
        print(f"Erro ao criar gráfico de candlestick para posição de longo prazo: {e}")
        import traceback
        traceback.print_exc()
        return False

# Função para criar gráfico detalhado da estratégia com candlesticks
def criar_grafico_estrategia(df, df_futuro, split_val, ticker, nome_ativo, setor, industria):
    try:
        print("Criando gráfico detalhado da estratégia...")
        
        # Configurar figura com 4 subplots
        fig = plt.figure(figsize=(15, 16))
        
        # 1. Gráfico de retorno acumulado
        ax1 = plt.subplot(4, 1, 1)
        df["retorno"].iloc[split_val:].cumsum().plot(label='Comprar e Manter', color='blue', linewidth=2)
        df["estrategia"].iloc[split_val:].cumsum().plot(label='Estratégia', color='green', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel("Data", fontsize=10)
        plt.ylabel("Retorno Acumulado", fontsize=10)
        plt.title(f"Comparação de Retorno Acumulado - {nome_ativo}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações explicativas
        retorno_acumulado = df["retorno"].iloc[split_val:].cumsum().iloc[-1]
        estrategia_acumulada = df["estrategia"].iloc[split_val:].cumsum().iloc[-1]
        
        if estrategia_acumulada > retorno_acumulado:
            plt.annotate(f'Estratégia superou Comprar e Manter\nDiferença: {(estrategia_acumulada-retorno_acumulado)*100:.2f}%',
                        xy=(df.index[split_val+len(df.iloc[split_val:])-1], estrategia_acumulada),
                        xytext=(df.index[split_val+int(len(df.iloc[split_val:])*0.8)], estrategia_acumulada*0.8),
                        arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                        fontsize=9)
        else:
            plt.annotate(f'Comprar e Manter superou Estratégia\nDiferença: {(retorno_acumulado-estrategia_acumulada)*100:.2f}%',
                        xy=(df.index[split_val+len(df.iloc[split_val:])-1], retorno_acumulado),
                        xytext=(df.index[split_val+int(len(df.iloc[split_val:])*0.8)], retorno_acumulado*0.8),
                        arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
                        fontsize=9)
        
        # 2. Gráfico de preço com candlesticks (últimos 60 dias)
        ax2 = plt.subplot(4, 1, 2)
        
        # Selecionar os últimos 60 dias para visualização
        ultimos_dias = 60
        df_recente = df.iloc[-ultimos_dias:].copy()
        
        # Plotar candlesticks
        for i in range(len(df_recente)):
            date = df_recente.index[i]
            op, hi, lo, cl = df_recente.iloc[i][['open', 'high', 'low', 'close']]
            
            # Determinar cor com base no preço de fechamento vs abertura
            color = 'green' if cl >= op else 'red'
            
            # Plotar corpo do candle
            ax2.plot([date, date], [op, cl], color=color, linewidth=4)
            
            # Plotar sombras
            ax2.plot([date, date], [lo, hi], color=color, linewidth=1)
        
        # Adicionar sinais de compra e venda
        sinais = np.sign(df_recente["prediction"].shift(1))
        datas_compra = df_recente.index[sinais > 0]
        datas_venda = df_recente.index[sinais < 0]
        
        ax2.scatter(datas_compra, df_recente.loc[datas_compra, "low"] * 0.99, color='green', marker='^', alpha=0.7, s=100, label='Compra')
        ax2.scatter(datas_venda, df_recente.loc[datas_venda, "high"] * 1.01, color='red', marker='v', alpha=0.7, s=100, label='Venda')
        
        # Adicionar médias móveis
        ax2.plot(df_recente.index, df_recente["SMA 15"], color='blue', linewidth=1, label='MM 15')
        ax2.plot(df_recente.index, df_recente["SMA 60"], color='orange', linewidth=1, label='MM 60')
        
        plt.xlabel("Data", fontsize=10)
        plt.ylabel("Preço (R$)", fontsize=10)
        plt.title(f"Candlesticks de {nome_ativo} com Sinais de Negociação", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 3. Gráfico de previsões futuras (3 meses) com candlesticks
        ax3 = plt.subplot(4, 1, 3)
        
        # Plotar os últimos 30 dias de dados reais com candlesticks
        ultimos_dias_reais = 30
        df_ultimos = df.iloc[-ultimos_dias_reais:].copy()
        
        for i in range(len(df_ultimos)):
            date = df_ultimos.index[i]
            op, hi, lo, cl = df_ultimos.iloc[i][['open', 'high', 'low', 'close']]
            
            # Determinar cor com base no preço de fechamento vs abertura
            color = 'green' if cl >= op else 'red'
            
            # Plotar corpo do candle
            ax3.plot([date, date], [op, cl], color=color, linewidth=4, alpha=0.7)
            
            # Plotar sombras
            ax3.plot([date, date], [lo, hi], color=color, linewidth=1, alpha=0.7)
        
        # Plotar as previsões futuras com candlesticks
        if df_futuro is not None:
            for i in range(len(df_futuro)):
                date = df_futuro.index[i]
                op, hi, lo, cl = df_futuro.iloc[i][['open', 'high', 'low', 'close']]
                
                # Determinar cor com base no preço de fechamento vs abertura
                color = 'green' if cl >= op else 'red'
                
                # Plotar corpo do candle com estilo diferente para indicar que é previsão
                ax3.plot([date, date], [op, cl], color=color, linewidth=4, alpha=0.4)
                
                # Plotar sombras
                ax3.plot([date, date], [lo, hi], color=color, linewidth=1, alpha=0.4)
            
            # Adicionar linha vertical para separar dados reais de previsões
            ax3.axvline(x=df.index[-1], color='black', linestyle='--', alpha=0.5)
            ax3.text(df.index[-1], df['close'].iloc[-1]*1.05, 'Início das Previsões', 
                    fontsize=9, ha='right', rotation=90, alpha=0.7)
            
            # Adicionar sinais de compra e venda nas previsões
            sinais_futuros = np.sign(df_futuro["retorno"])
            datas_compra_futuro = df_futuro.index[sinais_futuros > 0]
            datas_venda_futuro = df_futuro.index[sinais_futuros < 0]
            
            ax3.scatter(datas_compra_futuro, df_futuro.loc[datas_compra_futuro, "low"] * 0.99, 
                       color='green', marker='^', alpha=0.7, s=80, label='Sinal Compra Futuro')
            ax3.scatter(datas_venda_futuro, df_futuro.loc[datas_venda_futuro, "high"] * 1.01, 
                        color='red', marker='v', alpha=0.7, s=80, label='Sinal Venda Futuro')
        
        plt.xlabel("Data", fontsize=10)
        plt.ylabel("Preço (R$)", fontsize=10)
        plt.title(f"Previsão de Preços Futuros para {nome_ativo} (3 meses)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 4. Gráfico de indicadores técnicos relevantes
        ax4 = plt.subplot(4, 1, 4)
        
        # Plotar RSI
        plt.plot(df.index[-ultimos_dias:], df["RSI"].iloc[-ultimos_dias:], 
                label='RSI (14)', color='purple', linewidth=1.5)
        
        # Plotar MACD
        plt.plot(df.index[-ultimos_dias:], df["MACD"].iloc[-ultimos_dias:], 
                label='MACD', color='blue', linewidth=1.5)
        plt.plot(df.index[-ultimos_dias:], df["MACD_signal"].iloc[-ultimos_dias:], 
                label='Sinal MACD', color='red', linewidth=1.5)
        
        # Adicionar linhas de referência para RSI
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        plt.xlabel("Data", fontsize=10)
        plt.ylabel("Valor", fontsize=10)
        plt.title(f"Indicadores Técnicos para {nome_ativo}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Adicionar legenda explicativa
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Preço Histórico'),
            Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Previsão Futura'),
            Patch(facecolor='orange', alpha=0.2, label='Intervalo de Confiança (95%)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Sinal de Compra'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Sinal de Venda')
        ]
        
        # Adicionar explicação detalhada da estratégia
        plt.figtext(0.5, 0.01, 
                  "ESTRATÉGIAS DE NEGOCIAÇÃO PARA VALE S.A.\n\n"
                  f"Este gráfico compara duas estratégias de negociação para {nome_ativo} ({ticker}):\n"
                  f"Setor: {setor} | Indústria: {industria}\n\n"
                  "SWING TRADE (Curto Prazo):\n"
                  "• Utiliza modelo híbrido LSTM-Transformer-GRU para prever movimentos diários\n"
                  "• Opera em ambas direções (compra e venda) baseado nas previsões do modelo\n"
                  "• Ideal para capturar movimentos de curto prazo (1-5 dias)\n\n"
                  "POSIÇÃO DE LONGO PRAZO (Longo Prazo):\n"
                  "• Baseada em cruzamentos de médias móveis (MM 60 e MM 200)\n"
                  "• Mantém posições por períodos prolongados, ignorando ruídos de curto prazo\n"
                  "• Ideal para investidores com horizonte de tempo mais longo (meses ou anos)",
                  fontsize=10, ha='center', bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])
        plt.savefig(f'estrategia_detalhada_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo como 'estrategia_detalhada_{ticker}.png'")
        
        return True
    except Exception as e:
        print(f"Erro ao criar gráfico da estratégia: {e}")
        import traceback
        traceback.print_exc()
        return False

# Nova função para criar gráfico de candlestick para a estratégia de scalping
def criar_grafico_candlestick_scalping(df, df_futuro, ticker, nome_ativo):
    try:
        print("Criando gráfico de candlestick para estratégia de scalping...")
        
        # Selecionar os últimos 30 dias para visualização (foco em curto prazo para scalping)
        dias_visualizacao = 30
        df_recente = df.iloc[-dias_visualizacao:].copy()
        
        # Preparar dados para o gráfico de candlestick
        df_ohlc = df_recente[['open', 'high', 'low', 'close']].copy()
        
        # Adicionar volume
        df_ohlc['volume'] = df_recente['volume']
        
        # Adicionar médias móveis curtas (mais relevantes para scalping)
        df_ohlc['SMA5'] = df_recente['close'].rolling(5).mean()
        df_ohlc['SMA10'] = df_recente['close'].rolling(10).mean()
        
        # Identificar pontos de entrada para scalping baseados em cruzamento de médias curtas
        # e confirmação de RSI
        entradas = []
        stops = []
        alvos = []
        
        # Criar sinais de scalping baseados em cruzamento de médias e RSI
        for i in range(5, len(df_recente)-1):
            # Condição de entrada: SMA5 cruza acima da SMA10 e RSI > 50 (para compra)
            if (df_recente['SMA 15'].iloc[i-1] < df_recente['close'].rolling(10).mean().iloc[i-1] and 
                df_recente['SMA 15'].iloc[i] > df_recente['close'].rolling(10).mean().iloc[i] and 
                df_recente['RSI'].iloc[i] > 50):
                
                # Ponto de entrada
                entrada = df_recente['close'].iloc[i]
                entradas.append((df_recente.index[i], entrada))
                
                # Stop loss (1% abaixo do ponto de entrada)
                stop = entrada * 0.99
                stops.append((df_recente.index[i], stop))
                
                # Alvo (pelo menos 2x a distância do stop - relação 2:1)
                alvo = entrada + (entrada - stop) * 2
                alvos.append((df_recente.index[i], alvo))
        
        # Criar séries para plotagem
        entradas_x = [x[0] for x in entradas]
        entradas_y = [x[1] for x in entradas]
        
        stops_x = [x[0] for x in stops]
        stops_y = [x[1] for x in stops]
        
        alvos_x = [x[0] for x in alvos]
        alvos_y = [x[1] for x in alvos]
        
        # Preparar dados para mplfinance
        entradas_series = pd.Series(np.nan, index=df_recente.index)
        stops_series = pd.Series(np.nan, index=df_recente.index)
        alvos_series = pd.Series(np.nan, index=df_recente.index)
        
        for i, idx in enumerate(entradas_x):
            entradas_series.loc[idx] = entradas_y[i]
            stops_series.loc[idx] = stops_y[i]
            alvos_series.loc[idx] = alvos_y[i]
        
        # Adicionar anotações para os sinais
        apds = [
            mpf.make_addplot(df_ohlc['SMA5'], color='blue', width=1, label='Média Móvel 5'),
            mpf.make_addplot(df_ohlc['SMA10'], color='red', width=1, label='Média Móvel 10'),
            mpf.make_addplot(entradas_series, type='scatter', markersize=100, marker='^', color='green', label='Entrada'),
            mpf.make_addplot(stops_series, type='scatter', markersize=80, marker='_', color='red', label='Stop Loss'),
            mpf.make_addplot(alvos_series, type='scatter', markersize=80, marker='_', color='blue', label='Alvo (2:1)')
        ]
        
        # Configurar estilo do gráfico
        mc = mpf.make_marketcolors(
            up='green', down='red',
            edge='inherit',
            wick={'up':'green', 'down':'red'},
            volume='blue'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            figsize=(15, 10),
            gridstyle='--',
            y_on_right=False,
            volume_alpha=0.5
        )
        
        # Criar figura e salvar
        fig, axes = mpf.plot(
            df_ohlc,
            type='candle',
            style=s,
            addplot=apds,
            volume=True,
            panel_ratios=(4, 1),
            title=f'Estratégia de Scalping - {nome_ativo}',
            ylabel='Preço (R$)',
            ylabel_lower='Volume',
            returnfig=True
        )
        
        # Adicionar legenda
        axes[0].legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'scalping_candlestick_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo como 'scalping_candlestick_{ticker}.png'")
        
        # Criar imagem explicativa separada para a estratégia de scalping
        criar_imagem_explicativa_scalping(ticker)
        
        return True
    except Exception as e:
        print(f"Erro ao criar gráfico de candlestick para scalping: {e}")
        import traceback
        traceback.print_exc()
        return False

# Função para criar imagem explicativa da estratégia de scalping
def criar_imagem_explicativa_scalping(ticker):
    try:
        print("Criando imagem explicativa para estratégia de scalping...")
        
        # Criar figura
        fig = plt.figure(figsize=(12, 8))
        
        # Adicionar texto explicativo
        plt.text(0.5, 0.95, "ESTRATÉGIA DE SCALPING", fontsize=20, ha='center', weight='bold')
        
        plt.text(0.5, 0.85, 
                "O que é Scalping?", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.78, 
                "Scalping é uma estratégia de negociação de curtíssimo prazo que busca\n"
                "capturar pequenos movimentos de preço, muitas vezes em questão de minutos ou horas.\n"
                "O objetivo é realizar muitas operações com ganhos pequenos, mas consistentes.",
                fontsize=14, ha='center')
        
        plt.text(0.5, 0.65, 
                "Como funciona nossa estratégia de Scalping:", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.58, 
                "1. ENTRADA: Quando a média móvel de 5 períodos cruza acima da média de 10 períodos\n"
                "   e o RSI está acima de 50, indicando momentum positivo.\n\n"
                "2. STOP LOSS: Posicionado 1% abaixo do preço de entrada para limitar perdas.\n\n"
                "3. ALVO DE LUCRO: Definido com relação risco:retorno de pelo menos 2:1,\n"
                "   ou seja, se o risco é de 1%, o alvo será de pelo menos 2%.",
                fontsize=14, ha='center')
        
        plt.text(0.5, 0.35, 
                "Vantagens do Scalping:", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.28, 
                "• Exposição reduzida ao risco de mercado\n"
                "• Não mantém posições durante a noite\n"
                "• Potencial para ganhos consistentes em pequenos movimentos\n"
                "• Menos afetado por notícias e eventos de longo prazo",
                fontsize=14, ha='center')
        
        plt.text(0.5, 0.15, 
                "Desvantagens do Scalping:", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.08, 
                "• Requer atenção constante e tomada de decisão rápida\n"
                "• Custos de transação podem impactar significativamente os resultados\n"
                "• Exige disciplina rigorosa para seguir as regras de entrada e saída\n"
                "• Pode ser estressante devido ao ritmo acelerado das operações",
                fontsize=14, ha='center')
        
        # Remover eixos
        plt.axis('off')
        
        # Adicionar borda
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        
        # Salvar figura
        plt.savefig(f'explicacao_scalping_{ticker}.png', dpi=300, bbox_inches='tight', facecolor='lightyellow')
        plt.close()
        print(f"Imagem explicativa salva como 'explicacao_scalping_{ticker}.png'")
        
        return True
    except Exception as e:
        print(f"Erro ao criar imagem explicativa para scalping: {e}")
        import traceback
        traceback.print_exc()
        return False

# Função para criar imagens explicativas para as outras estratégias
def criar_imagens_explicativas(ticker):
    try:
        print("Criando imagens explicativas para as estratégias...")
        
        # Imagem explicativa para Swing Trade
        fig = plt.figure(figsize=(12, 8))
        
        plt.text(0.5, 0.95, "ESTRATÉGIA DE SWING TRADE", fontsize=20, ha='center', weight='bold')
        
        plt.text(0.5, 0.85, 
                "O que é Swing Trade?", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.78, 
                "Swing Trade é uma estratégia de negociação de curto a médio prazo que busca\n"
                "capturar movimentos de preço que duram de alguns dias a algumas semanas.\n"
                "O objetivo é aproveitar oscilações (swings) do mercado.",
                fontsize=14, ha='center')
        
        plt.text(0.5, 0.65, 
                "Como funciona nossa estratégia de Swing Trade:", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.58, 
                "1. ENTRADA: Baseada nas previsões do modelo híbrido LSTM-Transformer-GRU\n"
                "   que analisa padrões complexos nos dados e prevê a direção do movimento.\n\n"
                "2. DIREÇÃO: Opera em ambas direções (compra e venda) conforme as previsões.\n\n"
                "3. PERÍODO: Ideal para operações com duração de 1 a 5 dias.",
                fontsize=14, ha='center')
        
        plt.text(0.5, 0.35, 
                "Vantagens do Swing Trade:", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.28, 
                "• Menos estressante que day trade ou scalping\n"
                "• Requer menos tempo de monitoramento constante\n"
                "• Potencial para capturar movimentos maiores que o scalping\n"
                "• Menor impacto dos custos de transação",
                fontsize=14, ha='center')
        
        plt.text(0.5, 0.15, 
                "Desvantagens do Swing Trade:", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.08, 
                "• Exposição ao risco overnight\n"
                "• Vulnerabilidade a gaps e notícias inesperadas\n"
                "• Pode exigir maior capital para suportar volatilidade\n"
                "• Menor número de oportunidades comparado ao scalping",
                fontsize=14, ha='center')
        
        plt.axis('off')
        plt.savefig(f'explicacao_swing_trade_{ticker}.png', dpi=300, bbox_inches='tight', facecolor='lightyellow')
        plt.close()
        
        # Imagem explicativa para Holding Position
        fig = plt.figure(figsize=(12, 8))
        
        plt.text(0.5, 0.95, "ESTRATÉGIA DE POSIÇÃO DE LONGO PRAZO", fontsize=20, ha='center', weight='bold')
        
        plt.text(0.5, 0.85, 
                "O que é Posição de Longo Prazo?", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.78, 
                "A estratégia de posição de longo prazo (holding) consiste em manter um ativo\n"
                "por períodos prolongados, geralmente meses ou anos, ignorando flutuações de curto prazo\n"
                "e focando nas tendências fundamentais de longo prazo.",
                fontsize=14, ha='center')
        
        plt.text(0.5, 0.65, 
                "Como funciona nossa estratégia de Posição de Longo Prazo:", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.58, 
                "1. ENTRADA: Baseada no cruzamento da média móvel de 60 períodos acima da média de 200 períodos\n"
                "   (Golden Cross), indicando início de tendência de alta de longo prazo.\n\n"
                "2. SAÍDA: Quando a média de 60 períodos cruza abaixo da média de 200 períodos\n"
                "   (Death Cross), sinalizando possível reversão da tendência de alta.\n\n"
                "3. PERÍODO: Ideal para investidores com horizonte de tempo de meses ou anos.",
                fontsize=14, ha='center')
        
        plt.text(0.5, 0.35, 
                "Vantagens da Posição de Longo Prazo:", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.28, 
                "• Menor necessidade de monitoramento constante\n"
                "• Menor impacto dos custos de transação\n"
                "• Potencial para capturar grandes movimentos de tendência\n"
                "• Menos afetado por ruídos e volatilidade de curto prazo",
                fontsize=14, ha='center')
        
        plt.text(0.5, 0.15, 
                "Desvantagens da Posição de Longo Prazo:", 
                fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.08, 
                "• Capital fica comprometido por períodos mais longos\n"
                "• Exposição a riscos fundamentais e macroeconômicos\n"
                "• Pode perder oportunidades de curto prazo\n"
                "• Requer maior tolerância a drawdowns temporários",
                fontsize=14, ha='center')
        
        plt.axis('off')
        plt.savefig(f'explicacao_holding_{ticker}.png', dpi=300, bbox_inches='tight', facecolor='lightyellow')
        plt.close()
        
        print(f"Imagens explicativas salvas como 'explicacao_swing_trade_{ticker}.png' e 'explicacao_holding_{ticker}.png'")
        
        return True
    except Exception as e:
        print(f"Erro ao criar imagens explicativas: {e}")
        import traceback
        traceback.print_exc()
        return False

# Função para criar relatório com recomendações do modelo
# def criar_relatorio_recomendacoes(df, df_futuro, ticker, nome_ativo, mse, rmse, mae, r2):
#     try:
#         print("Criando relatório com recomendações do modelo...")
        
#         # Criar figura para o relatório
#         fig = plt.figure(figsize=(12, 16))
        
#         # Título
#         plt.text(0.5, 0.98, f"RELATÓRIO DE RECOMENDAÇÕES PARA {nome_ativo} ({ticker})", 
#                 fontsize=20, ha='center', weight='bold')
        
#         # Data do relatório
#         plt.text(0.5, 0.95, f"Gerado em: {datetime.datetime.now().strftime('%d/%m/%Y')}", 
#                 fontsize=14, ha='center')
        
#         # Métricas do modelo
#         plt.text(0.5, 0.91, "PRECISÃO DO MODELO:", fontsize=16, ha='center', weight='bold')
        
#         plt.text(0.5, 0.88, 
#                 f"Erro Quadrático Médio (MSE): {mse:.6f}\n"
#                 f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.6f}\n"
#                 f"Erro Absoluto Médio (MAE): {mae:.6f}\n"
#                 f"Coeficiente de Determinação (R²): {r2:.6f}",
#                 fontsize=14, ha='center')
        
#         # Interpretação da precisão
#         precisao_percentual = (1 - mae) * 100 if mae < 1 else 0
#         plt.text(0.5, 0.83, 
#                 f"Precisão estimada do modelo: {precisao_percentual:.2f}%\n"
#                 f"Margem de erro: ±{mae*100:.2f}%",
#                 fontsize=14, ha='center', color='blue')
        
#         # Linha divisória
#         plt.axhline(y=0.80, color='gray', linestyle='-', alpha=0.3)
        
#         # Recomendações para Scalping (próximo dia)
#         plt.text(0.5, 0.77, "RECOMENDAÇÕES PARA SCALPING (PRÓXIMO DIA):", 
#                 fontsize=16, ha='center', weight='bold')
        
#         # Obter previsão para o próximo dia
#         if df_futuro is not None and len(df_futuro) > 0:
#             prox_dia_retorno = df_futuro['retorno'].iloc[0]
#             prox_dia_preco = df_futuro['close'].iloc[0]
#             prox_dia_data = df_futuro.index[0].strftime('%d/%m/%Y')
            
#             if prox_dia_retorno > 0:
#                 recomendacao_scalp = "COMPRA"
#                 cor_scalp = "green"
#                 stop_loss = prox_dia_preco * 0.99
#                 alvo = prox_dia_preco + (prox_dia_preco - stop_loss) * 2
#             else:
#                 recomendacao_scalp = "VENDA"
#                 cor_scalp = "red"
#                 stop_loss = prox_dia_preco * 1.01
#                 alvo = prox_dia_preco - (stop_loss - prox_dia_preco) * 2
            
#             plt.text(0.5, 0.73, 
#                     f"Data: {prox_dia_data}\n"
#                     f"Preço previsto: R$ {prox_dia_preco:.2f}\n"
#                     f"Retorno previsto: {prox_dia_retorno*100:.2f}%\n"
#                     f"Recomendação: {recomendacao_scalp}",
#                     fontsize=14, ha='center', color=cor_scalp)
            
#             plt.text(0.5, 0.67, 
#                     f"Ponto de entrada: R$ {prox_dia_preco:.2f}\n"
#                     f"Stop loss: R$ {stop_loss:.2f} ({(stop_loss/prox_dia_preco-1)*100:.2f}%)\n"
#                     f"Alvo: R$ {alvo:.2f} ({(alvo/prox_dia_preco-1)*100:.2f}%)\n"
#                     f"Relação risco:retorno: 1:2",
#                     fontsize=14, ha='center')
#         else:
#             plt.text(0.5, 0.73, "Dados insuficientes para previsão de scalping", 
#                     fontsize=14, ha='center', color='gray')
        
#         # Linha divisória
#         plt.axhline(y=0.64, color='gray', linestyle='-', alpha=0.3)
        
#         # Recomendações para Swing Trade (próxima semana)
#         plt.text(0.5, 0.61, "RECOMENDAÇÕES PARA SWING TRADE (PRÓXIMA SEMANA):", 
#                 fontsize=16, ha='center', weight='bold')
        
#         # Obter previsão para a próxima semana (5 dias úteis)
#         if df_futuro is not None and len(df_futuro) >= 5:
#             semana_retorno = df_futuro['retorno'].iloc[:5].sum()
#             semana_preco_final = df_futuro['close'].iloc[4]
#             semana_data_final = df_futuro.index[4].strftime('%d/%m/%Y')
            
#             if semana_retorno > 0:
#                 recomendacao_swing = "COMPRA"
#                 cor_swing = "green"
#             else:
#                 recomendacao_swing = "VENDA"
#                 cor_swing = "red"
            
#             plt.text(0.5, 0.57, 
#                     f"Período: {df_futuro.index[0].strftime('%d/%m/%Y')} a {semana_data_final}\n"
#                     f"Preço final previsto: R$ {semana_preco_final:.2f}\n"
#                     f"Retorno acumulado previsto: {semana_retorno*100:.2f}%\n"
#                     f"Recomendação: {recomendacao_swing}",
#                     fontsize=14, ha='center', color=cor_swing)
            
#             # Adicionar detalhes dos dias da semana
#             dias_semana = []
#             for i in range(min(5, len(df_futuro))):
#                 dia = df_futuro.index[i].strftime('%d/%m')
#                 retorno = df_futuro['retorno'].iloc[i] * 100
#                 dias_semana.append(f"{dia}: {retorno:+.2f}%")
            
#             plt.text(0.5, 0.51, 
#                     "Previsão diária:\n" + "\n".join(dias_semana),
#                     fontsize=12, ha='center')
#         else:
#             plt.text(0.5, 0.57, "Dados insuficientes para previsão de swing trade", 
#                     fontsize=14, ha='center', color='gray')
        
#         # Linha divisória
#         plt.axhline(y=0.48, color='gray', linestyle='-', alpha=0.3)
        
#         # Recomendações para Holding (próximo ano)
#         plt.text(0.5, 0.45, "RECOMENDAÇÕES PARA POSIÇÃO DE LONGO PRAZO (PRÓXIMO ANO):", 
#                 fontsize=16, ha='center', weight='bold')
        
#         # Calcular tendência de longo prazo baseada nas médias móveis
#         ultimo_preco = df['close'].iloc[-1]
#         sma60 = df['SMA 60'].iloc[-1]
#         sma200 = df['SMA 200'].iloc[-1]
        
#         if sma60 > sma200:
#             tendencia = "ALTA"
#             recomendacao_holding = "COMPRA"
#             cor_holding = "green"
#         else:
#             tendencia = "BAIXA"
#             recomendacao_holding = "VENDA/AGUARDE"
#             cor_holding = "red"
        
#         # Projetar retorno anual baseado nas previsões disponíveis
#         if df_futuro is not None and len(df_futuro) > 0:
#             # Usar a média diária para projetar o ano
#             retorno_medio_diario = df_futuro['retorno'].mean()
#             retorno_anual_projetado = ((1 + retorno_medio_diario) ** 252) - 1
#             preco_anual_projetado = ultimo_preco * (1 + retorno_anual_projetado)
            
#             plt.text(0.5, 0.41, 
#                     f"Tendência atual: {tendencia}\n"
#                     f"Preço atual: R$ {ultimo_preco:.2f}\n"
#                     f"Preço projetado (1 ano): R$ {preco_anual_projetado:.2f}\n"
#                     f"Retorno anual projetado: {retorno_anual_projetado*100:.2f}%\n"
#                     f"Recomendação: {recomendacao_holding}",
#                     fontsize=14, ha='center', color=cor_holding)
            
#             # Adicionar análise técnica de longo prazo
#             plt.text(0.5, 0.34, 
#                     f"Análise técnica de longo prazo:\n"
#                     f"• Média Móvel 60 dias: R$ {sma60:.2f}\n"
#                     f"• Média Móvel 200 dias: R$ {sma200:.2f}\n"
#                     f"• Diferença entre médias: {(sma60/sma200-1)*100:+.2f}%",
#                     fontsize=14, ha='center')
#         else:
#             plt.text(0.5, 0.41, "Dados insuficientes para projeção de longo prazo", 
#                     fontsize=14, ha='center', color='gray')
        
#         # Linha divisória
#         plt.axhline(y=0.30, color='gray', linestyle='-', alpha=0.3)
        
#         # Resumo das recomendações
#         plt.text(0.5, 0.27, "RESUMO DAS RECOMENDAÇÕES:", fontsize=16, ha='center', weight='bold')
        
#         plt.text(0.5, 0.23, 
#                 f"Scalping (1 dia): {recomendacao_scalp if 'recomendacao_scalp' in locals() else 'N/A'}\n"
#                 f"Swing Trade (1 semana): {recomendacao_swing if 'recomendacao_swing' in locals() else 'N/A'}\n"
#                 f"Posição de Longo Prazo (1 ano): {recomendacao_holding}",
#                 fontsize=14, ha='center')
        
#         # Aviso importante
#         plt.text(0.5, 0.15, "AVISO IMPORTANTE:", fontsize=14, ha='center', weight='bold', color='red')
        
#         plt.text(0.5, 0.10, 
#                 "Este relatório é baseado em modelos matemáticos e análise técnica.\n"
#                 "Resultados passados não garantem retornos futuros.\n"
#                 "Sempre faça sua própria análise e considere seu perfil de risco\n"
#                 "antes de tomar decisões de investimento.",
#                 fontsize=12, ha='center', style='italic')
        
#         # Rodapé
#         plt.text(0.5, 0.03, 
#                 f"Análise gerada por modelo híbrido LSTM-Transformer-GRU\n"
#                 f"Precisão estimada: {precisao_percentual:.2f}%",
#                 fontsize=10, ha='center', color='gray')
        
#         # Remover eixos
#         plt.axis('off')
        
#         # Adicionar borda
#         plt.gca().spines['top'].set_visible(True)
#         plt.gca().spines['right'].set_visible(True)
#         plt.gca().spines['bottom'].set_visible(True)
#         plt.gca().spines['left'].set_visible(True)
        
#         # Salvar figura
#         plt.savefig(f'relatorio_recomendacoes_{ticker}.png', dpi=300, bbox_inches='tight', facecolor='white')
#         plt.close()
#         print(f"Relatório de recomendações salvo como 'relatorio_recomendacoes_{ticker}.png'")
        
#         return True
#     except Exception as e:
#         print(f"Erro ao criar relatório de recomendações: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
# Modificação na função criar_relatorio_recomendacoes
def criar_relatorio_recomendacoes(df, df_futuro, ticker, nome_ativo, mse, rmse, mae, r2):
    try:
        print("Criando relatório com recomendações do modelo...")
        
        # Criar figura para o relatório com mais espaço vertical
        fig = plt.figure(figsize=(12, 18))  # Aumentei a altura de 16 para 18
        
        # Título
        plt.text(0.5, 0.98, f"RELATÓRIO DE RECOMENDAÇÕES PARA {nome_ativo} ({ticker})", 
                fontsize=20, ha='center', weight='bold')
        
        # Data do relatório
        plt.text(0.5, 0.95, f"Gerado em: {datetime.datetime.now().strftime('%d/%m/%Y')}", 
                fontsize=14, ha='center')
        
        # Métricas do modelo
        plt.text(0.5, 0.91, "PRECISÃO DO MODELO:", fontsize=16, ha='center', weight='bold')
        
        plt.text(0.5, 0.88, 
                f"Erro Quadrático Médio (MSE): {mse:.6f}\n"
                f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.6f}\n"
                f"Erro Absoluto Médio (MAE): {mae:.6f}\n"
                f"Coeficiente de Determinação (R²): {r2:.6f}",
                fontsize=14, ha='center')
        
        # Interpretação da precisão
        precisao_percentual = (1 - mae) * 100 if mae < 1 else 0
        plt.text(0.5, 0.83, 
                f"Precisão estimada do modelo: {precisao_percentual:.2f}%\n"
                f"Margem de erro: ±{mae*100:.2f}%",
                fontsize=14, ha='center', color='blue')
        
        # Linha divisória
        plt.axhline(y=0.80, color='gray', linestyle='-', alpha=0.3)
        
        # Recomendações para Scalping (próximo dia)
        plt.text(0.5, 0.77, "RECOMENDAÇÕES PARA SCALPING (PRÓXIMO DIA):", 
                fontsize=16, ha='center', weight='bold')
        
        # Obter previsão para o próximo dia - SEMPRE USAR DADOS DISPONÍVEIS
        if df_futuro is not None and len(df_futuro) > 0:
            prox_dia_retorno = df_futuro['retorno'].iloc[0]
            prox_dia_preco = df_futuro['close'].iloc[0]
            prox_dia_data = df_futuro.index[0].strftime('%d/%m/%Y')
            
            if prox_dia_retorno > 0:
                recomendacao_scalp = "COMPRA"
                cor_scalp = "green"
                stop_loss = prox_dia_preco * 0.99
                alvo = prox_dia_preco + (prox_dia_preco - stop_loss) * 2
            else:
                recomendacao_scalp = "VENDA"
                cor_scalp = "red"
                stop_loss = prox_dia_preco * 1.01
                alvo = prox_dia_preco - (stop_loss - prox_dia_preco) * 2
            
            plt.text(0.5, 0.73, 
                    f"Data: {prox_dia_data}\n"
                    f"Preço previsto: R$ {prox_dia_preco:.2f}\n"
                    f"Retorno previsto: {prox_dia_retorno*100:.2f}%\n"
                    f"Recomendação: {recomendacao_scalp}",
                    fontsize=14, ha='center', color=cor_scalp)
            
            plt.text(0.5, 0.67, 
                    f"Ponto de entrada: R$ {prox_dia_preco:.2f}\n"
                    f"Stop loss: R$ {stop_loss:.2f} ({(stop_loss/prox_dia_preco-1)*100:.2f}%)\n"
                    f"Alvo: R$ {alvo:.2f} ({(alvo/prox_dia_preco-1)*100:.2f}%)\n"
                    f"Relação risco:retorno: 1:2",
                    fontsize=14, ha='center')
        else:
            # Usar dados históricos recentes se não houver previsões futuras
            ultimo_dia = df.iloc[-1]
            ultimo_dia_retorno = ultimo_dia['retorno']
            ultimo_dia_preco = ultimo_dia['close']
            ultimo_dia_data = df.index[-1].strftime('%d/%m/%Y')
            
            if ultimo_dia_retorno > 0:
                recomendacao_scalp = "COMPRA"
                cor_scalp = "green"
                stop_loss = ultimo_dia_preco * 0.99
                alvo = ultimo_dia_preco + (ultimo_dia_preco - stop_loss) * 2
            else:
                recomendacao_scalp = "VENDA"
                cor_scalp = "red"
                stop_loss = ultimo_dia_preco * 1.01
                alvo = ultimo_dia_preco - (stop_loss - ultimo_dia_preco) * 2
            
            plt.text(0.5, 0.73, 
                    f"Data: {ultimo_dia_data} (último dia disponível)\n"
                    f"Preço atual: R$ {ultimo_dia_preco:.2f}\n"
                    f"Retorno recente: {ultimo_dia_retorno*100:.2f}%\n"
                    f"Recomendação: {recomendacao_scalp}",
                    fontsize=14, ha='center', color=cor_scalp)
            
            plt.text(0.5, 0.67, 
                    f"Ponto de entrada: R$ {ultimo_dia_preco:.2f}\n"
                    f"Stop loss: R$ {stop_loss:.2f} ({(stop_loss/ultimo_dia_preco-1)*100:.2f}%)\n"
                    f"Alvo: R$ {alvo:.2f} ({(alvo/ultimo_dia_preco-1)*100:.2f}%)\n"
                    f"Relação risco:retorno: 1:2",
                    fontsize=14, ha='center')
        
        # Linha divisória - AUMENTAR ESPAÇAMENTO VERTICAL
        plt.axhline(y=0.63, color='gray', linestyle='-', alpha=0.3)  # Ajustado de 0.64 para 0.63
        
        # Recomendações para Swing Trade (próxima semana)
        plt.text(0.5, 0.60, "RECOMENDAÇÕES PARA SWING TRADE (PRÓXIMA SEMANA):", 
                fontsize=16, ha='center', weight='bold')  # Ajustado de 0.61 para 0.60
        
        # Obter previsão para a próxima semana (5 dias úteis)
        if df_futuro is not None and len(df_futuro) >= 5:
            semana_retorno = df_futuro['retorno'].iloc[:5].sum()
            semana_preco_final = df_futuro['close'].iloc[4]
            semana_data_final = df_futuro.index[4].strftime('%d/%m/%Y')
            
            if semana_retorno > 0:
                recomendacao_swing = "COMPRA"
                cor_swing = "green"
            else:
                recomendacao_swing = "VENDA"
                cor_swing = "red"
            
            plt.text(0.5, 0.56, 
                    f"Período: {df_futuro.index[0].strftime('%d/%m/%Y')} a {semana_data_final}\n"
                    f"Preço final previsto: R$ {semana_preco_final:.2f}\n"
                    f"Retorno acumulado previsto: {semana_retorno*100:.2f}%\n"
                    f"Recomendação: {recomendacao_swing}",
                    fontsize=14, ha='center', color=cor_swing)  # Ajustado de 0.57 para 0.56
            
            # Adicionar detalhes dos dias da semana
            dias_semana = []
            for i in range(min(5, len(df_futuro))):
                dia = df_futuro.index[i].strftime('%d/%m')
                retorno = df_futuro['retorno'].iloc[i] * 100
                dias_semana.append(f"{dia}: {retorno:+.2f}%")
            
            plt.text(0.5, 0.50, 
                    "Previsão diária:\n" + "\n".join(dias_semana),
                    fontsize=12, ha='center')  # Ajustado de 0.51 para 0.50
        else:
            # Usar dados históricos recentes se não houver previsões futuras
            ultimos_5_dias = df.iloc[-5:]
            semana_retorno = ultimos_5_dias['retorno'].sum()
            semana_preco_final = ultimos_5_dias['close'].iloc[-1]
            semana_data_inicial = ultimos_5_dias.index[0].strftime('%d/%m/%Y')
            semana_data_final = ultimos_5_dias.index[-1].strftime('%d/%m/%Y')
            
            if semana_retorno > 0:
                recomendacao_swing = "COMPRA"
                cor_swing = "green"
            else:
                recomendacao_swing = "VENDA"
                cor_swing = "red"
            
            plt.text(0.5, 0.56, 
                    f"Período: {semana_data_inicial} a {semana_data_final} (dados históricos)\n"
                    f"Preço final: R$ {semana_preco_final:.2f}\n"
                    f"Retorno acumulado recente: {semana_retorno*100:.2f}%\n"
                    f"Recomendação baseada em tendência recente: {recomendacao_swing}",
                    fontsize=14, ha='center', color=cor_swing)
            
            # Adicionar detalhes dos dias da semana
            dias_semana = []
            for i in range(len(ultimos_5_dias)):
                dia = ultimos_5_dias.index[i].strftime('%d/%m')
                retorno = ultimos_5_dias['retorno'].iloc[i] * 100
                dias_semana.append(f"{dia}: {retorno:+.2f}%")
            
            plt.text(0.5, 0.50, 
                    "Retornos diários recentes:\n" + "\n".join(dias_semana),
                    fontsize=12, ha='center')
        
        # Linha divisória - AUMENTAR ESPAÇAMENTO VERTICAL
        plt.axhline(y=0.46, color='gray', linestyle='-', alpha=0.3)  # Ajustado de 0.48 para 0.46
        
        # Recomendações para Holding (próximo ano)
        plt.text(0.5, 0.43, "RECOMENDAÇÕES PARA POSIÇÃO DE LONGO PRAZO (PRÓXIMO ANO):", 
                fontsize=16, ha='center', weight='bold')  # Ajustado de 0.45 para 0.43
        
        # Calcular tendência de longo prazo baseada nas médias móveis
        ultimo_preco = df['close'].iloc[-1]
        sma60 = df['SMA 60'].iloc[-1]
        sma200 = df['SMA 200'].iloc[-1]
        
        if sma60 > sma200:
            tendencia = "ALTA"
            recomendacao_holding = "COMPRA"
            cor_holding = "green"
        else:
            tendencia = "BAIXA"
            recomendacao_holding = "VENDA/AGUARDE"
            cor_holding = "red"
        
        # Projetar retorno anual baseado nas previsões disponíveis
        if df_futuro is not None and len(df_futuro) > 0:
            # Usar a média diária para projetar o ano
            retorno_medio_diario = df_futuro['retorno'].mean()
            retorno_anual_projetado = ((1 + retorno_medio_diario) ** 252) - 1
            preco_anual_projetado = ultimo_preco * (1 + retorno_anual_projetado)
            
            plt.text(0.5, 0.39, 
                    f"Tendência atual: {tendencia}\n"
                    f"Preço atual: R$ {ultimo_preco:.2f}\n"
                    f"Preço projetado (1 ano): R$ {preco_anual_projetado:.2f}\n"
                    f"Retorno anual projetado: {retorno_anual_projetado*100:.2f}%\n"
                    f"Recomendação: {recomendacao_holding}",
                    fontsize=14, ha='center', color=cor_holding)  # Ajustado de 0.41 para 0.39
            
            # Adicionar análise técnica de longo prazo
            plt.text(0.5, 0.32, 
                    f"Análise técnica de longo prazo:\n"
                    f"• Média Móvel 60 dias: R$ {sma60:.2f}\n"
                    f"• Média Móvel 200 dias: R$ {sma200:.2f}\n"
                    f"• Diferença entre médias: {(sma60/sma200-1)*100:+.2f}%",
                    fontsize=14, ha='center')  # Ajustado de 0.34 para 0.32
        else:
            # Usar dados históricos para projeção
            retorno_medio_diario = df['retorno'].iloc[-90:].mean()  # Média dos últimos 90 dias
            retorno_anual_projetado = ((1 + retorno_medio_diario) ** 252) - 1
            preco_anual_projetado = ultimo_preco * (1 + retorno_anual_projetado)
            
            plt.text(0.5, 0.39, 
                    f"Tendência atual: {tendencia}\n"
                    f"Preço atual: R$ {ultimo_preco:.2f}\n"
                    f"Preço projetado (1 ano): R$ {preco_anual_projetado:.2f}\n"
                    f"Retorno anual projetado: {retorno_anual_projetado*100:.2f}%\n"
                    f"Recomendação: {recomendacao_holding}",
                    fontsize=14, ha='center', color=cor_holding)
            
            # Adicionar análise técnica de longo prazo
            plt.text(0.5, 0.32, 
                    f"Análise técnica de longo prazo:\n"
                    f"• Média Móvel 60 dias: R$ {sma60:.2f}\n"
                    f"• Média Móvel 200 dias: R$ {sma200:.2f}\n"
                    f"• Diferença entre médias: {(sma60/sma200-1)*100:+.2f}%",
                    fontsize=14, ha='center')
        
        # Linha divisória - AUMENTAR ESPAÇAMENTO VERTICAL
        plt.axhline(y=0.28, color='gray', linestyle='-', alpha=0.3)  # Ajustado de 0.30 para 0.28
        
        # Resumo das recomendações
        plt.text(0.5, 0.25, "RESUMO DAS RECOMENDAÇÕES:", fontsize=16, ha='center', weight='bold')  # Ajustado de 0.27 para 0.25
        
        plt.text(0.5, 0.21, 
                f"Scalping (1 dia): {recomendacao_scalp if 'recomendacao_scalp' in locals() else 'N/A'}\n"
                f"Swing Trade (1 semana): {recomendacao_swing if 'recomendacao_swing' in locals() else 'N/A'}\n"
                f"Posição de Longo Prazo (1 ano): {recomendacao_holding}",
                fontsize=14, ha='center')  # Ajustado de 0.23 para 0.21
        
        # Aviso importante
        plt.text(0.5, 0.15, "AVISO IMPORTANTE:", fontsize=14, ha='center', weight='bold', color='red')
        
        plt.text(0.5, 0.10, 
                "Este relatório é baseado em modelos matemáticos e análise técnica.\n"
                "Resultados passados não garantem retornos futuros.\n"
                "Sempre faça sua própria análise e considere seu perfil de risco\n"
                "antes de tomar decisões de investimento.",
                fontsize=12, ha='center', style='italic')
        
        # Rodapé
        plt.text(0.5, 0.03, 
                f"Análise gerada por modelo híbrido LSTM-Transformer-GRU\n"
                f"Precisão estimada: {precisao_percentual:.2f}%",
                fontsize=10, ha='center', color='gray')
        
        # Remover eixos
        plt.axis('off')
        
        # Adicionar borda
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        
        # Salvar figura
        plt.savefig(f'relatorio_recomendacoes_{ticker}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Relatório de recomendações salvo como 'relatorio_recomendacoes_{ticker}.png'")
        
        return True
    except Exception as e:
        print(f"Erro ao criar relatório de recomendações: {e}")
        import traceback
        traceback.print_exc()
        return False

# Modificar a função principal para incluir as novas funcionalidades
def main():
    try:
        # Inicia o cronômetro
        start_time = time.time()
        
        # Verifica permissões de escrita
        try:
            with open('test_write.tmp', 'w') as f:
                f.write('test')
            os.remove('test_write.tmp')
            print("Diretório tem permissões de escrita.")
        except Exception as e:
            print(f"Aviso: Não foi possível escrever no diretório atual: {e}")
            print("Os gráficos e arquivos de saída podem não ser salvos corretamente.")
        
        # Extração dos dados (agora usando Yahoo Finance com o ticker correto)
        print("\n" + "="*50)
        print("ETAPA 1: EXTRAÇÃO DE DADOS DO YAHOO FINANCE")
        print("="*50)
        ticker = "VALE"  # Ticker correto da Vale na NYSE
        print(f"Baixando dados financeiros da Vale S.A. (NYSE: {ticker}) do Yahoo Finance...")
        df, nome_ativo, setor, industria = rmta_extrai_dados(ticker)
        print(f"Dados baixados com sucesso. Total de {len(df)} registros.")
        
        # O restante do código permanece igual até a parte das novas funcionalidades
        print("\n" + "="*50)
        print("ETAPA 2: ENGENHARIA DE ATRIBUTOS")
        print("="*50)
        print("Realizando engenharia de atributos...")
        df = rmta_func_engenharia_atributos(df)
        print(f"Engenharia de atributos concluída. Novas colunas: {list(df.columns)}")
        
        # Divisão dos dados
        print("\n" + "="*50)
        print("ETAPA 3: DIVISÃO DOS DADOS")
        print("="*50)
        print("Dividindo os dados em treino, validação e teste...")
        split = int(0.85 * len(df))
        split_val = int(0.95 * len(df))
        
        # Incluindo os novos indicadores na lista de features
        features = ['VWAP', 'RSI', 'SMA 15', 'SMA 60', 'SMA 200', 'MSD 15', 'MSD 60', 
                   'op', 'hi', 'lo', 'clo', 'vol', 'MACD', 'MACD_signal', 'MACD_diff', 
                   'BB_high', 'BB_low', 'BB_width', 'Stoch_K', 'Stoch_D', 'ATR', 'OBV',
                   'Ichimoku_A', 'Ichimoku_B', 'Price_Change', 'Price_Momentum',
                   'Volume_Change', 'Volume_MA_Ratio', 'Volatility', 'Trend_Strength', 'Trend_Direction']
        
        # Remover colunas categóricas para o modelo
        features_modelo = [f for f in features if f not in ['Day_of_Week', 'Month']]
        
        x_treino = df[features_modelo].iloc[:split,:]
        y_treino = df[['retorno']].iloc[:split,:]
        
        x_valid = df[features_modelo].iloc[split:split_val,:]
        y_valid = df[['retorno']].iloc[split:split_val,:]
        
        x_teste = df[features_modelo].iloc[split_val:,:]
        y_teste = df[['retorno']].iloc[split_val:,:]
        
        print(f"Divisão concluída: {len(x_treino)} amostras de treino, {len(x_valid)} de validação, {len(x_teste)} de teste")
        
        # Padronização
        print("\n" + "="*50)
        print("ETAPA 4: PADRONIZAÇÃO DOS DADOS")
        print("="*50)
        print("Padronizando os dados...")
        sc = StandardScaler()
        x_treino_sc = sc.fit_transform(x_treino)
        x_valid_sc = sc.transform(x_valid)
        x_teste_sc = sc.transform(x_teste)
        print("Padronização concluída.")
        
        # Ajuste no formato dos dados
        print("\n" + "="*50)
        print("ETAPA 5: AJUSTE DO FORMATO DOS DADOS")
        print("="*50)
        print("Ajustando o formato dos dados para o modelo...")
        lag = 15
        x_treino_final, y_treino_final = rmta_ajusta_formato_dados(x_treino_sc, y_treino.values, lag)
        x_valid_final, y_valid_final = rmta_ajusta_formato_dados(x_valid_sc, y_valid.values, lag)
        x_teste_final, y_teste_final = rmta_ajusta_formato_dados(x_teste_sc, y_teste.values, lag)
        
        print(f"Shape dos dados de treino: {x_treino_final.shape}")
        print(f"Shape dos dados de validação: {x_valid_final.shape}")
        print(f"Shape dos dados de teste: {x_teste_final.shape}")
        
        # Criação do modelo
        print("\n" + "="*50)
        print("ETAPA 6: CRIAÇÃO DO MODELO APRIMORADO")
        print("="*50)
        print("Criando o modelo Temporal Fusion Transformer aprimorado...")
        input_shape = x_treino_final.shape[1:]
        modelo_rmta = rmta_cria_modelo(input_shape,
                                    head_size=32,  # Aumentado
                                    num_heads=4,   # Aumentado
                                    ff_dim=16,     # Aumentado
                                    num_transformer_blocks=3,  # Aumentado
                                    mlp_units=256, # Aumentado
                                    dropout=0.2,   # Ajustado
                                    mlp_dropout=0.3)
        
        modelo_rmta.compile(loss="mean_squared_error", 
                         optimizer=keras.optimizers.Adam(learning_rate=0.001),
                         metrics=["mae"])  # Adicionado métrica MAE
        modelo_rmta.summary()
        
        # Treinamento do modelo com mais épocas
        print("\n" + "="*50)
        print("ETAPA 7: TREINAMENTO DO MODELO COM MAIS ÉPOCAS")
        print("="*50)
        print("Treinando o modelo com mais épocas...")
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),  # Aumentado
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=0.00005)  # Ajustado
        ]
        
        history = modelo_rmta.fit(x_treino_final,
                               y_treino_final,
                               validation_data=(x_valid_final, y_valid_final),
                               epochs=100,  # Aumentado para 100 épocas
                               batch_size=32,
                               callbacks=callbacks,
                               verbose=1)
        
        # Plot do histórico de treinamento com explicações
        print("\n" + "="*50)
        print("ETAPA 8: VISUALIZAÇÃO DO TREINAMENTO")
        print("="*50)
        print("Gerando gráfico do histórico de treinamento com explicações...")
        
        plt.figure(figsize=(15, 10))
        
        # Subplot para o erro de treinamento
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Treino', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validação', linewidth=2)
        plt.title(f'Evolução do Erro Durante o Treinamento - {nome_ativo}', fontsize=12)
        plt.xlabel('Época', fontsize=10)
        plt.ylabel('Erro Quadrático Médio', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True)
        
        # Adicionar anotações explicativas
        min_val_loss = min(history.history['val_loss'])
        min_val_loss_epoch = history.history['val_loss'].index(min_val_loss)
        plt.annotate(f'Melhor modelo: Época {min_val_loss_epoch+1}\nErro: {min_val_loss:.6f}',
                    xy=(min_val_loss_epoch, min_val_loss),
                    xytext=(min_val_loss_epoch+5, min_val_loss*1.2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=9)
        
        # Plot do MAE
        plt.subplot(2, 2, 2)
        plt.plot(history.history['mae'], label='MAE Treino', linewidth=2)
        plt.plot(history.history['val_mae'], label='MAE Validação', linewidth=2)
        plt.title('Erro Absoluto Médio Durante o Treinamento', fontsize=12)
        plt.xlabel('Época', fontsize=10)
        plt.ylabel('MAE', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True)
        
        # Plot da taxa de aprendizado
        if 'lr' in history.history:
            plt.subplot(2, 2, 3)
            plt.plot(history.history['lr'], linewidth=2)
            plt.title('Ajuste Automático da Taxa de Aprendizado', fontsize=12)
            plt.xlabel('Época', fontsize=10)
            plt.ylabel('Taxa de Aprendizado', fontsize=10)
            plt.grid(True)
            
            # Adicionar explicação
            plt.figtext(0.52, 0.35, 
                      "A taxa de aprendizado diminui automaticamente quando\n"
                      "o modelo para de melhorar, permitindo ajustes mais finos.",
                      fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        
        # Adicionar explicação geral do modelo
        plt.figtext(0.1, 0.02, 
                  f"MODELO DE PREVISÃO APRIMORADO PARA {nome_ativo.upper()}\n\n"
                  f"Este modelo híbrido combina redes neurais LSTM Bidirecionais, Transformer e GRU para capturar padrões temporais complexos nos dados financeiros.\n"
                  f"Foram utilizados {len(features_modelo)} indicadores técnicos como entrada, incluindo RSI, MACD, Bandas de Bollinger, Estocástico, ATR e outros.\n"
                  f"O modelo foi treinado com {len(x_treino)} amostras históricas e validado com {len(x_valid)} amostras independentes.\n"
                  f"O objetivo é prever a direção e magnitude do movimento do preço para o próximo dia de negociação.",
                  fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # Adicionar informações sobre o processo de treinamento
        plt.figtext(0.1, 0.45, 
                  "PROCESSO DE TREINAMENTO APRIMORADO:\n\n"
                  "1. Os dados foram divididos em conjuntos de treino (85%), validação (10%) e teste (5%).\n"
                  "2. Todos os indicadores foram padronizados para média 0 e desvio padrão 1.\n"
                  "3. O modelo usa uma janela de 15 dias para fazer previsões.\n"
                  "4. O treinamento foi configurado para até 100 épocas, com parada antecipada se não houver melhoria por 15 épocas.\n"
                  "5. A taxa de aprendizado foi reduzida quando o desempenho estagnou, permitindo ajustes mais precisos.",
                  fontsize=10, bbox=dict(facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(f'historico_treinamento_{ticker}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo como 'historico_treinamento_{ticker}.png'")
        
        # Previsões
        print("\n" + "="*50)
        print("ETAPA 9: PREVISÕES E AVALIAÇÃO")
        print("="*50)
        print("Fazendo previsões...")
        pred = modelo_rmta.predict(x_teste_final, verbose=0)
        
        # Avaliação do modelo
        mse = metrics.mean_squared_error(pred, y_teste_final)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(pred, y_teste_final)
        r2 = metrics.r2_score(y_teste_final, pred)
        
        print(f"Métricas de Avaliação:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        # Previsões e cálculo do retorno
        print("\n" + "="*50)
        print("ETAPA 10: CÁLCULO DE RETORNOS")
        print("="*50)
        print("Calculando retornos...")
        y_pred_treino = np.concatenate((np.zeros([lag,1]), modelo_rmta.predict(x_treino_final, verbose=0)), axis=0)
        y_pred_valid = np.concatenate((np.zeros([lag,1]), modelo_rmta.predict(x_valid_final, verbose=0)), axis=0)
        y_pred_teste = np.concatenate((np.zeros([lag,1]), modelo_rmta.predict(x_teste_final, verbose=0)), axis=0)
        
        df["prediction"] = np.concatenate((y_pred_treino, y_pred_valid, y_pred_teste), axis=0)
        
        # Cálculo da estratégia
        df["estrategia"] = df["retorno"] * np.sign(df["prediction"].shift(1))
        
        # Cálculo de métricas de trading
        retorno_acumulado = df["retorno"].iloc[split_val:].cumsum().iloc[-1]
        estrategia_acumulada = df["estrategia"].iloc[split_val:].cumsum().iloc[-1]
        
        print(f"Retorno acumulado comprar e manter (período de teste): {retorno_acumulado:.4f} ({retorno_acumulado*100:.2f}%")
        print(f"Retorno acumulado da estratégia (período de teste): {estrategia_acumulada:.4f} ({estrategia_acumulada*100:.2f}%")
        
        # Cálculo do Sharpe Ratio (assumindo taxa livre de risco de 0.1% ao dia)
        risk_free_rate = 0.001
        sharpe_bh = (df["retorno"].iloc[split_val:].mean() - risk_free_rate) / df["retorno"].iloc[split_val:].std()
        sharpe_strategy = (df["estrategia"].iloc[split_val:].mean() - risk_free_rate) / df["estrategia"].iloc[split_val:].std()
        
        print(f"Sharpe Ratio comprar e manter: {sharpe_bh:.4f}")
        print(f"Sharpe Ratio estratégia: {sharpe_strategy:.4f}")
        
        # Gerar previsões futuras para 3 meses (90 dias)
        print("\n" + "="*50)
        print("ETAPA 11: PREVISÕES FUTURAS PARA 3 MESES")
        print("="*50)
        dias_futuros = 90  # Prever os próximos 90 dias (3 meses)
        df_futuro = gerar_previsoes_futuras(modelo_rmta, df, features_modelo, sc, lag, dias_futuros)
        
        # Criar gráfico de barras para retornos mensais
        print("\n" + "="*50)
        print("ETAPA 12: CRIAÇÃO DE GRÁFICO DETALHADO DA ESTRATÉGIA")
        print("="*50)
        criar_grafico_estrategia(df, df_futuro, split_val, ticker, nome_ativo, setor, industria)
        
        # Criar gráficos específicos para swing trade e holding position
        print("\n" + "="*50)
        print("ETAPA 14: CRIAÇÃO DE GRÁFICOS ESPECÍFICOS PARA ESTRATÉGIAS")
        print("="*50)
        criar_grafico_candlestick_swing_trade(df, df_futuro, split_val, ticker, nome_ativo)
        criar_grafico_candlestick_holding(df, df_futuro, split_val, ticker, nome_ativo)
        
        # NOVA ETAPA: Criar gráfico de candlestick para estratégia de scalping
        print("\n" + "="*50)
        print("ETAPA 15: CRIAÇÃO DE GRÁFICO DE CANDLESTICK PARA ESTRATÉGIA DE SCALPING")
        print("="*50)
        criar_grafico_candlestick_scalping(df, df_futuro, ticker, nome_ativo)
        
        # NOVA ETAPA: Criar imagens explicativas para as estratégias
        print("\n" + "="*50)
        print("ETAPA 16: CRIAÇÃO DE IMAGENS EXPLICATIVAS PARA AS ESTRATÉGIAS")
        print("="*50)
        criar_imagens_explicativas(ticker)
        
        # NOVA ETAPA: Criar relatório com recomendações do modelo
        print("\n" + "="*50)
        print("ETAPA 17: CRIAÇÃO DE RELATÓRIO COM RECOMENDAÇÕES DO MODELO")
        print("="*50)
        criar_relatorio_recomendacoes(df, df_futuro, ticker, nome_ativo, mse, rmse, mae, r2)
        
        # Salvar o modelo
        print("\n" + "="*50)
        print("ETAPA 18: SALVANDO RESULTADOS")
        print("="*50)
        print("Salvando o modelo...")
        modelo_rmta.save(f'modelo_tft_{ticker}.h5')
        print(f"Modelo salvo como 'modelo_tft_{ticker}.h5'")
        
        # Salvar os resultados em CSV para análise posterior com explicações
        print("Salvando resultados em CSV...")
        
        # Adicionar uma coluna de sinal (1 para compra, -1 para venda, 0 para manter)
        df['sinal'] = np.sign(df["prediction"].shift(1))
        
        # Adicionar uma coluna de acerto (1 se o sinal estiver correto, 0 caso contrário)
        df['acerto'] = (np.sign(df["retorno"]) == np.sign(df["prediction"].shift(1))).astype(int)
        
        # Calcular o retorno acumulado
        df['retorno_acumulado'] = df["retorno"].cumsum()
        df['estrategia_acumulada'] = df["estrategia"].cumsum()
        
        # Selecionar as colunas relevantes
        resultados = df[['close', 'retorno', 'prediction', 'sinal', 'estrategia', 
                         'acerto', 'retorno_acumulado', 'estrategia_acumulada']].copy()
        
        # Adicionar previsões futuras ao CSV
        if df_futuro is not None:
            df_futuro['tipo'] = 'previsão'
            resultados['tipo'] = 'histórico'
            
            # Selecionar apenas as colunas compatíveis
            colunas_comuns = list(set(resultados.columns) & set(df_futuro.columns))
            resultados_futuros = pd.concat([resultados[colunas_comuns], df_futuro[colunas_comuns]])
        else:
            resultados_futuros = resultados
        
        # Salvar com cabeçalho explicativo detalhado
        with open(f'resultados_modelo_{ticker}.csv', 'w') as f:
            f.write(f"# Análise de {nome_ativo} - Gerado em {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"# Período analisado: {df.index[0].strftime('%d/%m/%Y')} a {df.index[-1].strftime('%d/%m/%Y')}\n")
            f.write(f"# Setor: {setor} | Indústria: {industria}\n")
            f.write(f"# Métricas do modelo: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}\n")
            f.write(f"# Retorno Comprar e Manter: {retorno_acumulado*100:.2f}%, Retorno Estratégia: {estrategia_acumulada*100:.2f}%\n")
            f.write(f"# Sharpe Ratio Comprar e Manter: {sharpe_bh:.4f}, Sharpe Ratio Estratégia: {sharpe_strategy:.4f}\n\n")
            
            f.write(f"# COMPARAÇÃO DE ESTRATÉGIAS DE NEGOCIAÇÃO\n")
            f.write(f"# Este arquivo contém resultados de três estratégias de negociação:\n")
            f.write(f"# 1. SCALPING: Operações de curtíssimo prazo com stop curto e relação risco:retorno de 2:1\n")
            f.write(f"# 2. SWING TRADE: Utiliza modelo híbrido LSTM-Transformer-GRU para prever movimentos diários\n")
            f.write(f"# 3. POSIÇÃO DE LONGO PRAZO: Baseada em cruzamentos de médias móveis de longo prazo\n\n")
            
            f.write(f"# Descrição das colunas:\n")
            f.write(f"# - close: Preço de fechamento em reais\n")
            f.write(f"# - retorno: Retorno diário real\n")
            f.write(f"# - prediction: Retorno previsto pelo modelo\n")
            f.write(f"# - sinal: Sinal de negociação (1=compra, -1=venda, 0=neutro)\n")
            f.write(f"# - estrategia: Retorno da estratégia (retorno * sinal)\n")
            f.write(f"# - acerto: Indica se o modelo acertou a direção (1=sim, 0=não)\n")
            f.write(f"# - retorno_acumulado: Retorno acumulado da estratégia Comprar e Manter\n")
            f.write(f"# - estrategia_acumulada: Retorno acumulado da estratégia do modelo\n")
            f.write(f"# - tipo: 'histórico' para dados reais, 'previsão' para previsões futuras\n\n")
        
        resultados_futuros.to_csv(f'resultados_modelo_{ticker}.csv', mode='a')
        print(f"Resultados salvos em 'resultados_modelo_{ticker}.csv'")
        
        # Tempo total de execução
        tempo_total = time.time() - start_time
        print("\n" + "="*50)
        print(f"PROCESSO CONCLUÍDO EM {tempo_total:.2f} SEGUNDOS")
        print("="*50)
        print(f"Análise de {nome_ativo} com dados do Yahoo Finance concluída com sucesso!")
        print(f"Arquivos gerados:")
        print(f"1. historico_treinamento_{ticker}.png - Gráfico do histórico de treinamento")
        print(f"2. retorno_mensal_barras_{ticker}.png - Gráfico de barras dos retornos mensais")
        print(f"3. estrategia_detalhada_{ticker}.png - Gráfico detalhado da estratégia com previsões de 3 meses")
        print(f"4. swing_trade_candlestick_{ticker}.png - Gráfico de candlestick para estratégia de swing trade")
        print(f"5. holding_position_candlestick_{ticker}.png - Gráfico de candlestick para estratégia de posição de longo prazo")
        print(f"6. scalping_candlestick_{ticker}.png - Gráfico de candlestick para estratégia de scalping")
        print(f"7. explicacao_scalping_{ticker}.png - Imagem explicativa da estratégia de scalping")
        print(f"8. explicacao_swing_trade_{ticker}.png - Imagem explicativa da estratégia de swing trade")
        print(f"9. explicacao_holding_{ticker}.png - Imagem explicativa da estratégia de posição de longo prazo")
        print(f"10. relatorio_recomendacoes_{ticker}.png - Relatório com recomendações do modelo")
        print(f"11. resultados_modelo_{ticker}.csv - Resultados detalhados e previsões futuras")
        print(f"12. modelo_tft_{ticker}.h5 - Modelo treinado salvo")
        
    except Exception as e:
        print(f"\nERRO DURANTE A EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

# Executa o programa principal
if __name__ == "__main__":
    print("Iniciando o programa...")
    exit_code = main()
    sys.exit(exit_code)
