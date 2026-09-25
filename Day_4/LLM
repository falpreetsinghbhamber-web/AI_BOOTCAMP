import random
from collections import defaultdict

# 1. Our "Training Data" (The LLM's dataset)
text = "the cat sat on the mat and the dog sat on the rug"
words = text.split()
words

# 2. The Model: Mapping a word to all the words that follow it
model = defaultdict(list)
for i in range(len(words) - 1):
    current_word = words[i]
    next_word = words[i + 1]
    model[current_word].append(next_word)
model

# 3. Text Generation: Start with a word and predict the next ones
current = "the"
output = [current]

for _ in range(5):
    # Look up possible next words and pick one at random
    possible_next_words = model[current]
    next_word = random.choice(possible_next_words)
    output.append(next_word)
    current = next_wordLL

print("Generated text:", " ".join(output))
