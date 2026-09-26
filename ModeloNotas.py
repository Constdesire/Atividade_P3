import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
n = 80

# base  de alunos
horas_estudo = np.random.uniform(1, 15, n)
frequencia = np.random.uniform(50, 100, n)
atividades_entregues = np.random.randint(0, 11, n)
nota_anterior = np.random.uniform(3, 10, n)


ruido = np.random.normal(0, 0.5, n)
nota_final = np.clip(10 * (
    0.25 * (horas_estudo / 15)
    + 0.25 * ((frequencia - 50) / 50)
    + 0.20 * (atividades_entregues / 10)
    + 0.30 * ((nota_anterior - 3) / 7)
) + ruido, 0, 10)

dados = pd.DataFrame({
    "horas_estudo": horas_estudo, "frequencia": frequencia,
    "atividades_entregues": atividades_entregues,
    "nota_anterior": nota_anterior, "nota_final": nota_final
})
dados.to_csv("base_alunos.csv", index=False)

# modelo de regressao linear
x = dados[["horas_estudo", "frequencia", "atividades_entregues", "nota_anterior"]]
y = dados["nota_final"]
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25, random_state=42)

modelo = LinearRegression().fit(x_treino, y_treino)
previsoes = np.clip(modelo.predict(x_teste), 0, 10)
erros = y_teste.values - previsoes

# calculos 
print("media das previsoes:", round(np.mean(previsoes), 2))
print("desvio padrao das previsoes:", round(np.std(previsoes, ddof=1), 2))
print("mae:", round(mean_absolute_error(y_teste, previsoes), 2))
print("rmse:", round(np.sqrt(mean_squared_error(y_teste, previsoes)), 2))
print("r2:", round(r2_score(y_teste, previsoes), 2))
media_erro, desvio_erro = np.mean(erros), np.std(erros, ddof=1)
print("media do erro:", round(media_erro, 2))
print("desvio padrao do erro:", round(desvio_erro, 2))
print("intervalo de erro 95%:", round(media_erro - 1.96 * desvio_erro, 2), "a", round(media_erro + 1.96 * desvio_erro, 2))


rosa, amarelo, rosa_claro = "#e75480", "#f2c14e", "#f6a5c0"

def salvar_grafico(nome):
    plt.tight_layout()
    plt.savefig(nome, dpi=150)
    plt.close()

plt.figure(figsize=(6, 4.5))
plt.scatter(dados["horas_estudo"], dados["nota_final"], color=rosa, edgecolor="white")
plt.title("Horas de estudo x Nota final")
plt.xlabel("Horas de estudo por semana")
plt.ylabel("Nota final")
salvar_grafico("grafico_horas_nota.png")

plt.figure(figsize=(6, 4.5))
plt.scatter(y_teste, previsoes, color=amarelo, edgecolor="black", s=60)
plt.plot([0, 10], [0, 10], color=rosa, linewidth=2)
plt.title("Nota real x Nota prevista pelo modelo")
plt.xlabel("Nota real")
plt.ylabel("Nota prevista")
salvar_grafico("grafico_real_previsto.png")

plt.figure(figsize=(6, 4.5))
plt.hist(erros, bins=8, color=rosa_claro, edgecolor=rosa)
plt.axvline(media_erro, color=amarelo, linewidth=2)
plt.title("Distribuição dos erros das previsões")
plt.xlabel("Erro (nota real menos nota prevista)")
plt.ylabel("Quantidade de alunos")
salvar_grafico("grafico_erros.png")

plt.figure(figsize=(6, 4.5))
nomes = ["Horas de\nestudo", "Frequência", "Atividades\nentregues", "Nota\nanterior"]
plt.bar(nomes, modelo.coef_, color=[rosa, amarelo, rosa_claro, rosa])
plt.title("Peso de cada característica no modelo")
plt.ylabel("Coeficiente")
salvar_grafico("grafico_coeficientes.png")

print("graficos e base de dados gerados")
