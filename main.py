import math

# FUNÇÃO DE ATIVAÇÃO DO NEURÔNIO

def sigmoid(x):
    return 1/(math.exp(-x))

x1 = 1
x2 = 2

w1 = 0.5
w2 = 0.9

bias = 0.1

# Soma ponderada
z = (x1 * w1) + (x2 * w2) + bias

# CHAMANDO A FUNÇÃO DE ATIVAÇÃO
output = sigmoid(z)

print(f"Saída do neurônio: {output:.4f}")