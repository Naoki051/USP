# -*- coding: utf-8 -*-
"""
Script completo e flexível para treinar, avaliar e gerar todos os artefatos
para uma CNN no dataset MNIST, cobrindo tarefas multiclasse e binária.
"""
# --- 1. Importação das Bibliotecas ---
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 2. Configuração dos Argumentos de Linha de Comando ---
def setup_parser():
    """Configura o parser de argumentos para total controle do experimento."""
    parser = argparse.ArgumentParser(description='Experimento completo de CNN no MNIST.')
    
    # Parâmetros da Tarefa e Execução
    parser.add_argument('--task', type=str, default='multiclass', choices=['multiclass', 'binary'], help='Tipo de tarefa: multiclasse ou binária.')
    parser.add_argument('--binary_digits', type=str, default='0,9', help='Quais dois dígitos usar para a tarefa binária (separados por vírgula). Ex: "4,9".')
    parser.add_argument('--output_dir', type=str, default='results', help='Diretório raiz para salvar os resultados.')
    
    # Parâmetros de Treinamento
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas de treinamento.')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do lote de treinamento.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Taxa de aprendizado do otimizador.')
    
    # Parâmetros da Arquitetura
    parser.add_argument('--conv_layers', type=int, default=2, choices=[1, 2], help='Número de camadas de convolução (1 ou 2).')
    parser.add_argument('--dense_units', type=int, default=128, help='Número de neurônios na camada densa.')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Taxa de dropout para regularização.')
    
    return parser.parse_args()

# --- 3. Funções de Dados e Modelo ---
def load_and_preprocess_data(task, binary_digits_str):
    """Carrega e pré-processa o MNIST de acordo com a tarefa (multiclasse ou binária)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalização e reformatação padrão
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    if task == 'binary':
        print(f"Filtrando dados para a tarefa binária com os dígitos: {binary_digits_str}")
        digits = [int(d) for d in binary_digits_str.split(',')]
        if len(digits) != 2:
            raise ValueError("Para a tarefa binária, especifique exatamente dois dígitos.")
        
        # Filtra os dados de treino
        train_mask = np.isin(y_train, digits)
        x_train, y_train = x_train[train_mask], y_train[train_mask]
        
        # Filtra os dados de teste
        test_mask = np.isin(y_test, digits)
        x_test, y_test = x_test[test_mask], y_test[test_mask]
        
        # Mapeia os dígitos para 0 e 1
        y_train = (y_train == digits[1]).astype(int)
        y_test = (y_test == digits[1]).astype(int)
        
        num_classes = 1
        # Para tarefa binária, não usamos one-hot encoding com binary_crossentropy
        y_train_cat, y_test_cat = y_train, y_test
    else: # multiclass
        num_classes = 10
        # Converte para one-hot encoding
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
        
    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat), num_classes


def build_cnn_model(input_shape, num_classes, args):
    """Constrói a arquitetura do modelo CNN com base nos argumentos e na tarefa."""
    model = Sequential(name=f"CNN_{args.task}")
    
    # Camadas convolucionais
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if args.conv_layers == 2:
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Camadas densas
    model.add(Flatten())
    model.add(Dense(args.dense_units, activation='relu'))
    model.add(Dropout(args.dropout_rate))
    
    # Camada de Saída
    if args.task == 'binary':
        model.add(Dense(1, activation='sigmoid'))
    else: # multiclass
        model.add(Dense(num_classes, activation='softmax'))
    
    return model

# --- 4. Funções de Avaliação e Plotagem ---
def evaluate_and_generate_artifacts(model, history, x_test, y_test, y_test_cat, run_dir, args):
    """Função central para avaliação e geração de todos os gráficos e relatórios."""
    print("\n--- Iniciando Avaliação e Geração de Artefatos ---")
    
    # Curvas de Aprendizado
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia do Modelo'); plt.xlabel('Época'); plt.ylabel('Acurácia'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treino')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda (Loss) do Modelo'); plt.xlabel('Época'); plt.ylabel('Perda'); plt.legend()
    plt.savefig(os.path.join(run_dir, 'curvas_aprendizado.png')); plt.close()
    print("✅ Gráfico de curvas de aprendizado salvo.")

    # Previsões
    y_pred_prob = model.predict(x_test)
    
    if args.task == 'binary':
        y_pred_classes = (y_pred_prob > 0.5).astype("int32").flatten()
        target_names = [f'Dígito {d}' for d in args.binary_digits.split(',')]
    else:
        y_pred_classes = np.argmax(y_pred_prob, axis=1)
        target_names = [str(i) for i in range(10)]

    # Relatório de Classificação
    report = classification_report(y_test, y_pred_classes, target_names=target_names)
    with open(os.path.join(run_dir, 'relatorio_classificacao.txt'), 'w') as f:
        f.write(report)
    print("✅ Relatório de classificação salvo.")
    print("\nRelatório de Classificação:\n", report)

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusão'); plt.ylabel('Classe Verdadeira'); plt.xlabel('Classe Prevista')
    plt.savefig(os.path.join(run_dir, 'matriz_confusao.png')); plt.close()
    print("✅ Matriz de confusão salva.")
    
    # Curva ROC (apenas para tarefa binária)
    if args.task == 'binary':
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taxa de Falsos Positivos'); plt.ylabel('Taxa de Verdadeiros Positivos'); plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(run_dir, 'curva_roc.png')); plt.close()
        print(f"✅ Curva ROC com AUC = {auc:.4f} salva.")

    # Salvando previsões
    pd.DataFrame({'rotulo_real': y_test, 'rotulo_previsto': y_pred_classes}).to_csv(os.path.join(run_dir, 'previsoes_teste.csv'), index=False)
    print("✅ Previsões do teste salvas.")


# --- 5. Função Principal de Execução ---
def main():
    """Função principal que orquestra todo o processo do experimento."""
    args = setup_parser()
    
    # Cria um diretório único para esta execução baseado no tempo
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{timestamp}_{args.task}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Resultados serão salvos em: '{run_dir}'")

    # Salva os hiperparâmetros
    with open(os.path.join(run_dir, 'hiperparametros.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Carrega os dados de acordo com a tarefa
    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat), num_classes = load_and_preprocess_data(args.task, args.binary_digits)
    input_shape = x_train.shape[1:]

    # Constrói o modelo
    model = build_cnn_model(input_shape, num_classes if args.task == 'multiclass' else 1, args)
    
    # Compila o modelo de acordo com a tarefa
    loss_function = 'binary_crossentropy' if args.task == 'binary' else 'categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    
    # Salva o resumo do modelo em um arquivo
    with open(os.path.join(run_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    model.summary()
    
    # Salva os pesos iniciais
    model.save_weights(os.path.join(run_dir, 'pesos_iniciais.weights.h5'))
    
    # Configura o logger para salvar o histórico do treino
    csv_logger = CSVLogger(os.path.join(run_dir, 'log_treinamento.csv'), append=False)
    
    # Treina o modelo e mede o tempo
    print(f"\n🚀 Iniciando treinamento para a tarefa '{args.task}'...")
    start_time = time.time()
    history = model.fit(
        x_train, y_train_cat,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test_cat),
        callbacks=[csv_logger],
        verbose=1
    )
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"✅ Treinamento concluído em {training_duration:.2f} segundos.")
    with open(os.path.join(run_dir, 'tempo_treinamento.txt'), 'w') as f:
        f.write(f"Tempo total de treinamento: {training_duration:.2f} segundos\n")

    # Salva os pesos finais
    model.save_weights(os.path.join(run_dir, 'pesos_finais.weights.h5'))
    
    # Avalia e gera todos os artefatos de saída
    evaluate_and_generate_artifacts(model, history, x_test, y_test, y_test_cat, run_dir, args)

    print("\n🎉 Experimento finalizado com sucesso!")

# --- Ponto de Entrada do Script ---
if __name__ == '__main__':
    main()