import numpy as np
import pandas as pd
import PCA.utilsPCA as acp
from PCA import graphicsPCA as g


def AnalizaComponentePrincipale():

    # citirea si salvarea datelor intr-un dataframe
    tabel = pd.read_csv('dataIN/Date_AmericaLatina.csv', index_col=0)

    # extragere lista de variabile utile
    numeVar = tabel.columns.values
    print(numeVar, type(numeVar))

    # creare lista etichete observatii
    numeObs = tabel.index.values
    print(numeObs, type(numeObs))

    # nr. de variabile
    m = len(numeVar)
    print(m)
    # nr. de observatii
    n = numeObs.shape[0]
    print(n)

    # extragere matrice observatii-variabile cauzale
    X = tabel[numeVar].values
    print(X, X.shape, type(X))

    # standardizare matrice variabile cauzale
    X_std = acp.standardise(X)
    print(X_std.shape, type(X_std))
    X_std_df = pd.DataFrame(data=X_std,
                            index=numeObs,
                            columns=numeVar)
    # salvare matrice standardizata
    X_std_df.to_csv('PCA/dataOUT/X_standardizat.csv')

    # instantiere clasa ACP
    modelACP = acp.PCA(X_std)
    alpha = modelACP.getAlpha()
    print("alpha=", alpha)

    # creare grafic explicare a variantei
    g.componentePrincipale(valoriProprii=alpha)
    # g.afisare()

    # extragere componente principale
    compPrin = modelACP.getComponentePrincipale()
    componente = ['C'+str(j+1) for j in range(compPrin.shape[1])]
    compPrin_df = pd.DataFrame(data=compPrin, index=numeObs,
                               columns=componente)
    compPrin_df.to_csv('PCA/dataOUT/ComponentePrincipale.csv')
    # compPrin_df.to_excel('./dataOUT/ComponentePrincipale.xlsx')

    # extragere factori de corelatie (factor loadings)
    Rxc = modelACP.getFactorLoadings()
    Rxc_df = pd.DataFrame(data=Rxc, index=numeVar, columns=componente)
    Rxc_df.to_csv("PCA/dataOUT/FactoriCorelatie.csv")
    g.corelograma(matrice=Rxc_df, titlu='Corelograma factorilor de corelatie')
    # g.afisare()

    # calcul scoruri
    scoruri = modelACP.getScoruri()
    scoruri_df = pd.DataFrame(data=scoruri, index=numeObs,
                              columns=componente)
    scoruri_df.to_csv("PCA/dataOUT/Scoruri.csv")
    g.harta_intensitate(matrice=scoruri_df, titlu='Componente principale standardizate (scoruri)')
    # g.afisare()

    # calcul calitatea reprezentarii observatiilor
    calObs = modelACP.getCalitateObservatii()
    calObs_df = pd.DataFrame(data=calObs, index=numeObs, columns=componente)
    calObs_df.to_csv("PCA/dataOUT/CalitateObs.csv")
    g.harta_intensitate(matrice=calObs_df, titlu='Calitatea reprezentarii observatiilor')
    # g.afisare()

    # extragere contributie observatii la explicarea variantei axelor componentelor principale
    contribObs = modelACP.getContributieObservatii()
    contribObs_df = pd.DataFrame(data=contribObs, index=numeObs, columns=componente)
    contribObs_df.to_csv("PCA/dataOUT/ContributieObs.csv")
    g.harta_intensitate(matrice=contribObs_df, titlu='Contributia observatiilor la varianta axelor')
    # g.afisare()

    # extragere comunalitati (componentele principale regasite in variabilele initiale)
    comun = modelACP.getComunalitati()
    comun_df = pd.DataFrame(data=comun, index=numeVar, columns=componente)
    # salvare in fisier CSV
    comun_df.to_csv("PCA/dataOUT/Comunalitati.csv")
    g.harta_intensitate(matrice=comun_df, titlu='Harta comunalitatilor')
    # g.afisare()

    # crearea cercului corelatiilor pentru evidentierea corelatiei dintre
    # variabilele initiale si C1, C2
    g.cerculCorelatiilor(matrice=Rxc_df, titlu='Corelatia dintre variabilele initiale si C1, C2')

    # variabilele initiale si C3, C4
    g.cerculCorelatiilor(matrice=Rxc_df, X1=2, X2=3, titlu='Corelatia dintre variabilele initiale si C3, C4')

    # crearea cercului corelatiilor pentru evidentierea legaturii dintre observatii si C1, C2
    maxim_scor = np.max(scoruri)
    minim_scor = np.min(scoruri)
    print('Maxim scor, folosit ca raza pentru cercul corelatiilor: ', maxim_scor)
    g.cerculCorelatiilor(matrice=scoruri_df, raza=maxim_scor, valMin=minim_scor, valMax=maxim_scor,
                         titlu='Distributia observatiilor in spatiul C1, C2')

    g.cerculCorelatiilor(matrice=scoruri_df, raza=maxim_scor, valMin=minim_scor, valMax=maxim_scor, X1=2, X2=3,
                         titlu='Distributia observatiilor in spatiul C3, C4')

    g.afisare()
