# ====================================================================
# PROJETO DA DISCIPLINA DE ESTATÍSTICA - Mestrado CIn 2019.2
# Tema: Tetes de hipótese sobre a base do Enem 2018
# ====================================================================

# Importando as bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

#------------------------------
# ANÁLISE EXPLORATÓRIA DE DADOS
#------------------------------

# Carregando o dataset
print("Carregando o dataset...\n")
df = pd.read_csv("D:\Thiago\Mestrado\Base_Enem_Reduzida1.csv")

# Exibindo as primeiras 5 linhas
print("Head do dataset")
print(df.head())

# Exibindo informações gerais sobre o dataset
print("\nInformações gerais sobre o dataset")
df.info()

print("\nRealizando ajustes no dataset...")

# Alterando o tipo da coluna NU_IDADE
df["NU_IDADE"] = df["NU_IDADE"].astype("int32")

# Alterando o tipo da coluna TP_SEXO
df["TP_SEXO"] = df["TP_SEXO"].astype("category")

# Criando uma coluna com a média das notas
df["media_notas"] = df[["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT"]].mean(axis=1)

# Excluindo as colunas que foram sumarizadas
df = df.drop(columns=["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT"])

print(df.shape)

# Exibindo informações estatísticas
print("\nDados estatísticos das colunas numméricas")
print(df.describe())

# Média das notas por tipo de escola
media_tipoescola = df.groupby("Tipo_Escola")["media_notas"].mean()
print("\nMédia por tipo de escola")
print(media_tipoescola)
media_tipoescola.nsmallest(4).plot.barh(title="Média das notas por tipo de escola", color="r")
plt.xlabel("Média notas")
#plt.show()

# Média das notas por faixa de renda
renda = df.groupby("Renda_Mensal")["media_notas"].mean()
print("\nMédia por renda")
print(renda)
#%matplotlib inline
renda.nsmallest(17).plot.barh(title="Média das notas por Renda", color="r")
plt.xlabel("Média Notas")
#plt.show()

# Média das notas por escolaridade do pai
escolaridade_pai = df.groupby("Escolaridade_Pai")["media_notas"].mean()
print("\nMédia por escolaridade do pai")
print(escolaridade_pai)
escolaridade_pai.nsmallest(9).plot.barh(title="Média das notas por Escolaridade do Pai", color="r")
plt.xlabel("Média Notas")
#plt.show()

# Média das notas por escolaridade da mãe
escolaridade_mae = df.groupby("Escolaridade_mae")["media_notas"].mean()
print("\nMédia por escolaridade da mãe")
print(escolaridade_mae)
escolaridade_mae.nsmallest(9).plot.barh(title="Média das notas por Escolaridade da Mãe", color="r")
plt.xlabel("Média Notas")
#plt.show()

# Exibindo as 10 maiores médias
print("\nAs 10 maiores médias")
print(df.nlargest(10, "media_notas"))

# Exibindo as 10 menores médias
print("\nAs 10 menores médias")
print(df.nsmallest(10, "media_notas"))

# Diferença nas médias entre escolas públicas e privadas
escola_publica = df.loc[df["Tipo_Escola"] == "Pública"]
escola_privada = df.loc[df["Tipo_Escola"] == "Privada"]
dif_medias = escola_publica["media_notas"].mean() - escola_privada["media_notas"].mean()
print("\nDiferença das médias entre escolas públicas e privadas")
print(dif_medias)

#---------------------
# TESTES DE HIPÓTESES
#---------------------

# Hipótese 1: Alunos de escolas privadas têm melhor desempenho do que os alunos de escolas públicas
# H0: Não existe diferença de desempenho
# H1: Existe diferença de desempenho
# a: 5%
N1 = escola_publica["media_notas"]
N2 = escola_privada["media_notas"]
dif_medias = escola_publica["media_notas"].mean() - escola_privada["media_notas"].mean()

# Aplicando o t-teste
resultado_teste = sm.stats.ttest_ind(x1=N1,x2=N2, value=0, alternative="two-sided", weights=(None, None),usevar='pooled')
print("\nResultado do t-teste")
print(resultado_teste)
if resultado_teste[1] <= 0.05:
    print("Rejeita-se a hipótese nula")
else:
    print("Falha em rejeitar a Hipótese nula")

resultado_teste = sm.stats.ztest(x1=N1, x2=N2, value=0, alternative="two-sided", usevar='pooled')
print("\nResultado do z-teste")
print(resultado_teste)
