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
    print("Execute 'pip install ta scikit-learn pandas numpy tensorflow yfinance matplotlib' para instalar as dependências")
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
                label='Buy & Hold', 
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
                label='Buy & Hold Acumulado')
        
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
                  "ESTRATÉGIA DE NEGOCIAÇÃO DIRECIONAL\n\n"
                  "Esta estratégia utiliza aprendizado profundo para prever a direção do movimento de preço da ação no próximo dia.\n"
                  "Quando o modelo prevê um retorno positivo, a estratégia assume uma posição comprada (LONG).\n"
                  "Quando o modelo prevê um retorno negativo, a estratégia assume uma posição vendida (SHORT).\n"
                  "O desempenho é medido comparando os retornos acumulados da estratégia com a abordagem passiva de Buy & Hold.",
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

# Função para criar gráfico detalhado da estratégia
def criar_grafico_estrategia(df, df_futuro, split_val, ticker, nome_ativo, setor, industria):
    try:
        print("Criando gráfico detalhado da estratégia...")
        
        # Configurar figura com 4 subplots
        fig = plt.figure(figsize=(15, 16))
        
        # 1. Gráfico de retorno acumulado
        ax1 = plt.subplot(4, 1, 1)
        df["retorno"].iloc[split_val:].cumsum().plot(label='Buy & Hold', color='blue', linewidth=2)
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
            plt.annotate(f'Estratégia superou Buy & Hold\nDiferença: {(estrategia_acumulada-retorno_acumulado)*100:.2f}%',
                        xy=(df.index[split_val+len(df.iloc[split_val:])-1], estrategia_acumulada),
                        xytext=(df.index[split_val+int(len(df.iloc[split_val:])*0.8)], estrategia_acumulada*0.8),
                        arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                        fontsize=9)
        else:
            plt.annotate(f'Buy & Hold superou Estratégia\nDiferença: {(retorno_acumulado-estrategia_acumulada)*100:.2f}%',
                        xy=(df.index[split_val+len(df.iloc[split_val:])-1], retorno_acumulado),
                        xytext=(df.index[split_val+int(len(df.iloc[split_val:])*0.8)], retorno_acumulado*0.8),
                        arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
                        fontsize=9)
        
        # 2. Gráfico de preço com sinais de compra e venda
        ax2 = plt.subplot(4, 1, 2)
        plt.plot(df.index[split_val:], df["close"].iloc[split_val:], label='Preço Real', color='blue', linewidth=2)
        
        # Adicionar sinais de compra e venda
        sinais = np.sign(df["prediction"].shift(1)).iloc[split_val:]
        datas_compra = df.index[split_val:][sinais > 0]
        datas_venda = df.index[split_val:][sinais < 0]
        
        plt.scatter(datas_compra, df.loc[datas_compra, "close"], color='green', marker='^', alpha=0.7, s=50, label='Compra')
        plt.scatter(datas_venda, df.loc[datas_venda, "close"], color='red', marker='v', alpha=0.7, s=50, label='Venda')
        
        plt.xlabel("Data", fontsize=10)
        plt.ylabel("Preço ($)", fontsize=10)
        plt.title(f"Preço de {nome_ativo} com Sinais de Negociação", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 3. Gráfico de previsões futuras (3 meses)
        ax3 = plt.subplot(4, 1, 3)
        
        # Plotar os últimos 60 dias de dados reais
        ultimos_dias = 60
        plt.plot(df.index[-ultimos_dias:], df["close"].iloc[-ultimos_dias:], 
                label='Preço Histórico', color='blue', linewidth=2)
        
        # Plotar as previsões futuras
        if df_futuro is not None:
            plt.plot(df_futuro.index, df_futuro["close"], 
                    label='Previsão Futura (3 meses)', color='orange', linewidth=2, linestyle='--')
            
            # Adicionar intervalo de confiança (simplificado)
            std_retorno = df["retorno"].std()
            plt.fill_between(df_futuro.index, 
                            df_futuro["close"] * (1 - 1.96 * std_retorno), 
                            df_futuro["close"] * (1 + 1.96 * std_retorno),
                            color='orange', alpha=0.2, label='Intervalo de Confiança (95%)')
            
            # Adicionar sinais de compra e venda nas previsões
            sinais_futuros = np.sign(df_futuro["retorno"])
            datas_compra_futuro = df_futuro.index[sinais_futuros > 0]
            datas_venda_futuro = df_futuro.index[sinais_futuros < 0]
            
            plt.scatter(datas_compra_futuro, df_futuro.loc[datas_compra_futuro, "close"], 
                       color='green', marker='^', alpha=0.7, s=50, label='Sinal Compra Futuro')
            plt.scatter(datas_venda_futuro, df_futuro.loc[datas_venda_futuro, "close"], 
                        color='red', marker='v', alpha=0.7, s=50, label='Sinal Venda Futuro')
        
        plt.xlabel("Data", fontsize=10)
        plt.ylabel("Preço ($)", fontsize=10)
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
                  "ESTRATÉGIA DE NEGOCIAÇÃO DIRECIONAL COM APRENDIZADO PROFUNDO\n\n"
                  f"Esta estratégia utiliza um modelo híbrido LSTM-Transformer-GRU para prever a direção do movimento de preço de {nome_ativo} ({ticker}).\n"
                  f"Setor: {setor} | Indústria: {industria}\n\n"
                  "FUNCIONAMENTO DA ESTRATÉGIA:\n"
                  "1. O modelo analisa 15 dias de histórico de preços e indicadores técnicos para prever o retorno do próximo dia.\n"
                  "2. Quando o modelo prevê um retorno positivo, a estratégia assume uma posição comprada (LONG).\n"
                  "3. Quando o modelo prevê um retorno negativo, a estratégia assume uma posição vendida (SHORT).\n"
                  "4. As posições são ajustadas diariamente com base nas novas previsões do modelo.\n\n"
                  "VANTAGENS DESTA ABORDAGEM:\n"
                  "• Captura movimentos em ambas as direções do mercado (alta e baixa)\n"
                  "• Adapta-se a diferentes condições de mercado através do aprendizado profundo\n"
                  "• Utiliza múltiplos indicadores técnicos para identificar padrões complexos\n"
                  "• Fornece previsões de preço para os próximos 3 meses para planejamento de longo prazo",
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

# Função principal que executa todo o processo
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
        
        # O restante do código permanece igual
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
        
        print(f"Retorno acumulado buy & hold (período de teste): {retorno_acumulado:.4f} ({retorno_acumulado*100:.2f}%)")
        print(f"Retorno acumulado da estratégia (período de teste): {estrategia_acumulada:.4f} ({estrategia_acumulada*100:.2f}%)")
        
        # Cálculo do Sharpe Ratio (assumindo taxa livre de risco de 0.1% ao dia)
        risk_free_rate = 0.001
        sharpe_bh = (df["retorno"].iloc[split_val:].mean() - risk_free_rate) / df["retorno"].iloc[split_val:].std()
        sharpe_strategy = (df["estrategia"].iloc[split_val:].mean() - risk_free_rate) / df["estrategia"].iloc[split_val:].std()
        
        print(f"Sharpe Ratio buy & hold: {sharpe_bh:.4f}")
        print(f"Sharpe Ratio estratégia: {sharpe_strategy:.4f}")
        
        # Gerar previsões futuras para 3 meses (90 dias)
        print("\n" + "="*50)
        print("ETAPA 11: PREVISÕES FUTURAS PARA 3 MESES")
        print("="*50)
        dias_futuros = 90  # Prever os próximos 90 dias (3 meses)
        df_futuro = gerar_previsoes_futuras(modelo_rmta, df, features_modelo, sc, lag, dias_futuros)
        
        # Criar gráfico de barras para retornos mensais
        print("\n" + "="*50)
        print("ETAPA 12: CRIAÇÃO DE GRÁFICO DE BARRAS PARA RETORNOS MENSAIS")
        print("="*50)
        criar_grafico_barras_retorno_mensal(df, split_val, ticker, nome_ativo)
        
        # Criar gráfico detalhado da estratégia
        print("\n" + "="*50)
        print("ETAPA 13: CRIAÇÃO DE GRÁFICO DETALHADO DA ESTRATÉGIA")
        print("="*50)
        criar_grafico_estrategia(df, df_futuro, split_val, ticker, nome_ativo, setor, industria)
        
        # Salvar o modelo
        print("\n" + "="*50)
        print("ETAPA 14: SALVANDO RESULTADOS")
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
            f.write(f"# Retorno Buy & Hold: {retorno_acumulado*100:.2f}%, Retorno Estratégia: {estrategia_acumulada*100:.2f}%\n")
            f.write(f"# Sharpe Ratio Buy & Hold: {sharpe_bh:.4f}, Sharpe Ratio Estratégia: {sharpe_strategy:.4f}\n\n")
            
            f.write(f"# ESTRATÉGIA DE NEGOCIAÇÃO DIRECIONAL COM APRENDIZADO PROFUNDO\n")
            f.write(f"# Esta estratégia utiliza um modelo híbrido LSTM-Transformer-GRU para prever a direção do movimento de preço.\n")
            f.write(f"# Quando o modelo prevê um retorno positivo, a estratégia assume uma posição comprada (LONG).\n")
            f.write(f"# Quando o modelo prevê um retorno negativo, a estratégia assume uma posição vendida (SHORT).\n")
            f.write(f"# As posições são ajustadas diariamente com base nas novas previsões do modelo.\n\n")
            
            f.write(f"# Descrição das colunas:\n")
            f.write(f"# - close: Preço de fechamento em dólares\n")
            f.write(f"# - retorno: Retorno diário real\n")
            f.write(f"# - prediction: Retorno previsto pelo modelo\n")
            f.write(f"# - sinal: Sinal de negociação (1=compra, -1=venda, 0=neutro)\n")
            f.write(f"# - estrategia: Retorno da estratégia (retorno * sinal)\n")
            f.write(f"# - acerto: Indica se o modelo acertou a direção (1=sim, 0=não)\n")
            f.write(f"# - retorno_acumulado: Retorno acumulado da estratégia Buy & Hold\n")
            f.write(f"# - estrategia_acumulada: Retorno acumulado da estratégia do modelo\n")
            f.write(f"# - tipo: 'histórico' para dados reais, 'previsão' para previsões futuras (3 meses)\n\n")
        
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
        print(f"4. resultados_modelo_{ticker}.csv - Resultados detalhados e previsões futuras")
        print(f"5. modelo_tft_{ticker}.h5 - Modelo treinado salvo")
        
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
