try:
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print("Biblioteki zostały zainportowane")
except:
    print("Nie udało się zaimportować bibliotek")
    

debug = 1

#default cpu
device = torch.device("cpu")

# GPU
try:
        import torch_directml # type: ignore
        device = torch_directml.device()
        print("Program korzysta z AMD GPU przez DirectML")
except ImportError:
        print("torch_directml nie jest zainstalowany, przełączam na CPU")
        device = torch.device("cpu")


if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Program korzysta z NVIDIA GPU")
else:
        device = torch.device("cpu")
        print("CUDA niedostępne, program korzysta z CPU")


#DANE
curr_dir = os.getcwd()
if (debug == 1):
    path = os.path.join(curr_dir, "word2vec", "test.txt")
else:
    path = os.path.join(curr_dir, "word2vec", "turbomini.txt")
#


if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("lis skacze nad płotem a pies goni lisa")

with open(path, "r", encoding="utf-8") as file:
    text = file.read().lower().split()

# Słowniki
vocab = sorted(list(set(text))) #tworzy listę oryginalnych słów
word_to_idx = {word: i for i, word in enumerate(vocab)} #zwraca (i,word)
idx_to_word = {i: word for word, i in word_to_idx.items()} #zwraca (word,i)
vocab_size = len(vocab)

T = len(text)
m = 2 #kontekst
data = []

# Tworzenie par treningowych (Skip-gram)
for k in range(T):
    for l in range(-m, m + 1):
        if l == 0:
            continue

        context_pos = k + l
        if 0 <= context_pos < T:
            target = word_to_idx[text[k]]
            context = word_to_idx[text[context_pos]]
            data.append((target, context))

# --- 3. ARCHITEKTURA MODELU ---
class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.output_layer = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.output_layer(x)
        return x

embedding_dim = 10
model = SkipGram(vocab_size, embedding_dim).to(device) #Ładuje dane na gpu

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# --- 4. PĘTLA TRENINGOWA ---
print(f"Rozpoczynam trening na {device}...")
epochs = 100

for epoch in range(epochs):
    total_loss = 0.0

    for target, context in data:
        target_t = torch.tensor([target], dtype=torch.long, device=device)
        context_t = torch.tensor([context], dtype=torch.long, device=device)

        optimizer.zero_grad()
        output = model(target_t)
        loss = criterion(output, context_t)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / len(data)
        print(f"Epoka {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        
        
        
        
#===============================
# REZULTAT
#===============================


def get_vector(word):
    if word not in word_to_idx:
        raise ValueError(f"Słowo '{word}' nie istnieje w słowniku.")
    idx = torch.tensor([word_to_idx[word]], dtype=torch.long, device=device)
    with torch.no_grad():
        return model.embeddings(idx).squeeze(0)

def find_most_similar(word, n=3):
    if word not in word_to_idx:
        print(f"Brak słowa '{word}' w słowniku.")
        return

    with torch.no_grad():
        target_v = get_vector(word)                     # [emb_dim]
        all_weights = model.embeddings.weight          # [vocab_size, emb_dim]

        sims = F.cosine_similarity(
            target_v.unsqueeze(0),                     # [1, emb_dim]
            all_weights,                               # [vocab_size, emb_dim]
            dim=1
        )

        # weź więcej niż n, bo pierwsze zwykle będzie to samo słowo
        vals, indices = torch.topk(sims, min(n + 1, vocab_size))

        print(f"Słowa najbardziej podobne do '{word}':")
        found = 0
        for score, idx in zip(vals, indices):
            idx = idx.item()
            similar_word = idx_to_word[idx]

            if similar_word == word:
                continue

            print(f" -> {similar_word} ({score.item():.4f})")
            found += 1

            if found >= n:
                break

# Przykład
while True:
    word = input("Podaj słowo")
    find_most_similar(word)