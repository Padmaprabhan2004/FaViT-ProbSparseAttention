import numpy as np
import matplotlib.pyplot as plt

def get_sinusoidal_pe(max_len, d_model):
    position=np.arange(max_len)[:, np.newaxis]
    div_term=np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe =np.zeros((max_len, d_model))
    pe[:, 0::2]=np.sin(position * div_term)
    pe[:, 1::2]=np.cos(position * div_term)
    return pe


max_len = 100
d_model = 512
n_trials = 25000
i = 10          

pe = get_sinusoidal_pe(max_len, d_model)
pi = pe[i]
pj = pe[i+1:]  


avg_scores = np.zeros(max_len - i - 1)

np.random.seed(42)
for _ in range(n_trials):
    W = np.random.randn(d_model, d_model) * (1 / np.sqrt(d_model))  
    Wpj = pj @ W.T       
    scores = Wpj @ pi   
    avg_scores += scores

avg_scores /= n_trials  ###avg 

plt.figure(figsize=(10, 5))
plt.plot(range(i+1, max_len), avg_scores, marker='o', linestyle='-')
plt.title(r"Averaged $p_i^T W p_j$ vs $j$ over {} random $W$ matrices".format(n_trials), fontsize=14)
plt.xlabel("Position j", fontsize=12)
plt.ylabel(r"$\mathbb{E}_W[p_i^T W p_j]$", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("pos_embedding_decay_plot.jpg",dpi=300)
plt.show()
