#====================================================================
# PROJETO DA DISCIPLINA DE ESTATÍSTICA - Mestrado CIn 2019.2
# Tema: Tetes de hipótese sobre a base do Enem 2018
#====================================================================

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
df = pd.read_csv("D:\Thiago\Mestrado\Base_Enem_Reduzida3_Estados.csv")

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
df["media_notas"] = df[["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]].mean(axis=1)

# Excluindo as colunas desnecessárias
df = df.drop(columns=["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"])

print(df.shape)

# Exibindo informações estatísticas
print("\nDados estatísticos das colunas numméricas")
print(df.describe())

# Média das notas por tipo de escola
media_tipoescola = df.groupby("Tipo_Escola")["media_notas"].mean()
print("\nMédia por tipo de escola: ", media_tipoescola)
media_tipoescola.nsmallest(4).plot.barh(title="Média das notas por tipo de escola", color="r")
plt.xlabel("Média notas")
plt.show()
plt.hist(x=media_tipoescola, orientation='vertical')
plt.show()

# Média das notas por faixa de renda
renda = df.groupby("Renda_Mensal")["media_notas"].mean()
print("\nMédia por renda")
print(renda.nsmallest(17))
#%matplotlib inline
renda.nsmallest(17).plot.barh(title="Média das notas por Renda", color="r")
plt.xlabel("Média Notas")
plt.show()
plt.hist(x=renda, orientation='vertical')
plt.show()

# Média das notas por escolaridade do pai
escolaridade_pai = df.groupby("Escolaridade_Pai")["media_notas"].mean()
print("\nMédia por escolaridade do pai")
print(escolaridade_pai)
escolaridade_pai.nsmallest(9).plot.barh(title="Média das notas por Escolaridade do Pai", color="r")
plt.xlabel("Média Notas")
plt.show()
plt.hist(x=escolaridade_pai, orientation='vertical')
plt.show()

# Média das notas por escolaridade da mãe
escolaridade_mae = df.groupby("Escolaridade_mae")["media_notas"].mean()
print("\nMédia por escolaridade da mãe")
print(escolaridade_mae)
escolaridade_mae.nsmallest(9).plot.barh(title="Média das notas por Escolaridade da Mãe", color="r")
plt.xlabel("Média Notas")
plt.show()
plt.hist(x=escolaridade_mae, orientation='vertical')
plt.show()

# Exibindo as 10 maiores médias
print("\nAs 10 maiores médias")
print(df.nlargest(10, "media_notas"))

# Exibindo as 10 menores médias
print("\nAs 10 menores médias")
print(df.nsmallest(10, "media_notas"))

#---------------------
# TESTES DE HIPÓTESES
#---------------------

# Hipótese 1: Alunos de escolas privadas têm melhor desempenho do que os alunos de escolas públicas
# H0: Não existe diferença de desempenho
# H1: Existe diferença de desempenho
# a: 5%
escola_publica = df.loc[df["Tipo_Escola"] == "Pública"]
escola_privada = df.loc[df["Tipo_Escola"] == "Privada"]
dif_medias = escola_publica["media_notas"].mean() - escola_privada["media_notas"].mean()
print("\nDiferença das médias entre escolas públicas e privadas: ", dif_medias)

# Aplicando o t-teste
N1 = escola_publica["media_notas"]
N2 = escola_privada["media_notas"]
resultado_teste = sm.stats.ttest_ind(x1=N1,x2=N2, value=0, alternative="two-sided", weights=(None, None),usevar='pooled')
print("\nResultado do t-teste: ", resultado_teste)

# Conclusão sobre o resultado
if resultado_teste[1] <= 0.05:
    print("Rejeita-se a hipótese nula")
else:
    print("Falha em rejeitar a Hipótese nula")

sns.boxplot(x=df["Tipo_Escola"], y=df["media_notas"], data=df)

# Hipótese 2: Alunos de classe média têm melhor desempenho do que os alunos de classe baixa
# H0: Não existe diferença de desempenho
# H1: Existe diferença de desempenho
# a: 5%
baixa_renda = df.loc[df["Renda_Mensal"].isin(["Nenhuma renda", "Até R$ 954,00", "De R$ 954,01 até R$ 1.431,00", "De R$ 1.431,01 até R$ 1.908,00",
                                              "De R$ 1.908,01 até R$ 2.385,00", "De R$ 2.385,01 até R$ 2.862,00"])]
print("\nshape baixa renda:", baixa_renda.shape)

classe_media = df.loc[df["Renda_Mensal"].isin(["De R$ 2.862,01 até R$ 3.816,00", "De R$ 3.816,01 até R$ 4.770,00", "De R$ 4.770,01 até R$ 5.724,00",
                                               "De R$ 5.724,01 até R$ 6.678,00"])]
print("shape classe média:", classe_media.shape)

classe_alta = df.loc[df["Renda_Mensal"].isin(["De R$ 6.678,01 até R$ 7.632,00", "De R$ 7.632,01 até R$ 8.586,00", "De R$ 8.586,01 até R$ 9.540,00",
                                              "De R$ 9.540,01 até R$ 11.448,00", "De R$ 11.448,01 até R$ 14.310,00"])]
print("shape classe alta:", classe_alta.shape)

dif_medias = baixa_renda["media_notas"].mean() - classe_media["media_notas"].mean()
print("\nDiferença das médias entre baixa renda e classe média: ", dif_medias)

dif_medias = classe_media["media_notas"].mean() - classe_alta["media_notas"].mean()
print("\nDiferença das médias entre classe média e classe alta: ", dif_medias)

# Aplicando o t-teste
N1 = baixa_renda["media_notas"]
N2 = classe_media["media_notas"]
resultado_teste = sm.stats.ttest_ind(x1=N1,x2=N2, value=0, alternative="two-sided", weights=(None, None),usevar='pooled')
print("\nResultado do t-teste (baixa renda X classe média): ", resultado_teste)

# Conclusão sobre o resultado
if resultado_teste[1] <= 0.05:
    print("Rejeita-se a hipótese nula")
else:
    print("Falha em rejeitar a Hipótese nula")

N1 = classe_media["media_notas"]
N2 = classe_alta["media_notas"]
resultado_teste = sm.stats.ttest_ind(x1=N1,x2=N2, value=0, alternative="two-sided", weights=(None, None),usevar='pooled')
print("\nResultado do t-teste (classe média X classe alta): ", resultado_teste)

# Conclusão sobre o resultado
if resultado_teste[1] <= 0.05:
    print("Rejeita-se a hipótese nula")
else:
    print("Falha em rejeitar a Hipótese nula")

# Hipótese 3: Alunos cujos pais têm maior nível de escolaridade têm melhor desempenho
# H0: Não existe diferença de desempenho
# H1: Existe diferença de desempenho
# a: 5%
escolaridade_baixa = df.query("(Escolaridade_Pai == 'Nunca estudou' or " +
                               "Escolaridade_Pai == 'Não completou a 4ª série/5º ano do Ensino Fundamental' or " +
                               "Escolaridade_Pai == 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental') and " +
                              "(Escolaridade_mae == 'Nunca estudou' or " +
                               "Escolaridade_mae == 'Não completou a 4ª série/5º ano do Ensino Fundamental' or " +
                               "Escolaridade_mae == 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental')")
print("\nshape escolaridade baixa:", escolaridade_baixa.shape)

escolaridade_media = df.query("(Escolaridade_Pai == 'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio' or " +
                               "Escolaridade_Pai == 'Completou o Ensino Médio, mas não completou a Faculdade') or " +
                              "(Escolaridade_mae == 'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio' or " +
                               "Escolaridade_mae == 'Completou o Ensino Médio, mas não completou a Faculdade')")
print("shape escolaridade média:", escolaridade_media.shape)

escolaridade_alta = df.query("(Escolaridade_Pai == 'Completou a Faculdade, mas não completou a Pós-graduação' or " +
                               "Escolaridade_Pai == 'Completou a Pós-graduação') or " +
                              "(Escolaridade_mae == 'Completou a Faculdade, mas não completou a Pós-graduação' or " +
                               "Escolaridade_mae == 'Completou a Pós-graduação')")
print("shape escolaridade alta:", escolaridade_alta.shape)

dif_medias = escolaridade_baixa["media_notas"].mean() - escolaridade_media["media_notas"].mean()
print("\nDiferença das médias entre escolaridade dos pais baixa e média: ", dif_medias)

dif_medias = escolaridade_media["media_notas"].mean() - escolaridade_alta["media_notas"].mean()
print("\nDiferença das médias entre escolaridade dos pais média e alta: ", dif_medias)

# Aplicando o t-teste
N1 = escolaridade_baixa["media_notas"]
N2 = escolaridade_media["media_notas"]
resultado_teste = sm.stats.ttest_ind(x1=N1,x2=N2, value=0, alternative="two-sided", weights=(None, None),usevar='pooled')
print("\nResultado do t-teste (escolaridade baixa X escolaridade média): ", resultado_teste)

# Conclusão sobre o resultado
if resultado_teste[1] <= 0.05:
    print("Rejeita-se a hipótese nula")
else:
    print("Falha em rejeitar a Hipótese nula")

N1 = escolaridade_media["media_notas"]
N2 = escolaridade_alta["media_notas"]
resultado_teste = sm.stats.ttest_ind(x1=N1,x2=N2, value=0, alternative="two-sided", weights=(None, None),usevar='pooled')
print("\nResultado do t-teste (escolaridade média X escolaridade alta): ", resultado_teste)

# Conclusão sobre o resultado
if resultado_teste[1] <= 0.05:
    print("Rejeita-se a hipótese nula")
else:
    print("Falha em rejeitar a Hipótese nula")