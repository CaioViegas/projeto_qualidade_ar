import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.pipeline import Pipeline

def preparar_dados(df, coluna_alvo):
    X = df.drop(columns=[coluna_alvo], axis=1)
    y = df[coluna_alvo]
    return X, y

def criar_pipeline():
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('rf', RandomForestClassifier(random_state=101))
    ])
    return pipeline

def otimizar_modelo(X_train, y_train, pipeline, parametros):
    grid_search = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    return grid_search.best_estimator_

def calcular_metricas(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def executar_pipeline(df, coluna_alvo, parametros, caminho_salvar=None):
    X, y = preparar_dados(df, coluna_alvo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    pipeline = criar_pipeline()
    modelo_otimizado = otimizar_modelo(X_train, y_train, pipeline=pipeline, parametros=parametros)
    cv_scores = cross_val_score(modelo_otimizado, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Pontuação de Cross-Validation (R2): {cv_scores}")
    y_pred = modelo_otimizado.predict(X_test)
    accuracy, precision, recall, f1 = calcular_metricas(y_test, y_pred)
    resultados = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    df_resultados = pd.DataFrame([resultados])
    print("\nResultados do modelo:")
    print(df_resultados)

    if caminho_salvar:
        joblib.dump(modelo_otimizado, caminho_salvar)
        print(f"\nModelo salvo em {caminho_salvar}")

if __name__ == "__main__":
    param_grid = {
        'rf__n_estimators': [50, 100, 150, 200],  
        'rf__max_depth': [5, 15, 30],       
        'rf__min_samples_split': [2, 5, 10],       
        'rf__min_samples_leaf': [1, 2, 4],         
        'rf__max_features': ['sqrt', 'log2'],      
        'rf__bootstrap': [True, False]             
    }
    df = pd.read_csv("./Data/dataset_codificado.csv")
    caminho_salvar = "./Modelo/random_forest_otimizado.joblib"
    executar_pipeline(df, "Qualidade_Ar", param_grid, caminho_salvar=caminho_salvar)