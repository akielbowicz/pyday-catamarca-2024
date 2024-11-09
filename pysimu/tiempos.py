from timeit import repeat
from experimento import experimento

def main(tamanios):
    resultados = dict()
    for t in tamanios:
        print(f"Calculando tiempos para grillas de tamaño: {t}")
        r = repeat("paso(ci,pi)", setup=f"paso, ci, pi = experimento({t})", repeat=20, number=100, globals=globals())
        resultados[t] = r
        
    return resultados

if __name__ == "__main__":
    tamanios = [8,16,32,50, ]#64,100, 128, 256]
    main(tamanios)