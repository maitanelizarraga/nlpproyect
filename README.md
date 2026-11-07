# nlpproyect
Para qué es cada .py:
  python main.py                              # E2 completo + E3 completo
  python main.py --solo-e2                    # solo E2 (necesario la primera vez: regenera el Word2Vec corregido)
  python main.py --solo-e3 --rapido           # prueba de humo del E3 (usa lo que el E2 ya guardó)
  python main.py --solo-e3 --fase 1 --sin-deep   # fase 1 solo con shallow (minutos, sin GPU)
  python main.py --solo-e3 --sin-bert         # E3 sin RoBERTa
  python main.py --solo-e3                    # E3 completo (GPU recomendable)
Salidas del E3 en resultados_e3/.