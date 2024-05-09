import faiss
import numpy as np

text_chunks = [''' Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been
used successfully in a variety of tasks including reading comprehension, abstractive summarization,
textual entailment and learning task-independent sentence representations''', 
'''End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned
recurrence and have been shown to perform well on simple-language question answering and
language modeling tasks''', '''We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of
queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the
query with all keys, divide each by
√
dk, and apply a softmax function to obtain the weights on the
values.''']  # Replace with your sample texts

dimension = 10  # Set a dummy dimension for the example
embeddings = np.random.rand(len(text_chunks), dimension).astype('float32')

vectorstore = faiss.IndexFlatL2(dimension)
vectorstore.add(embeddings)
print("Vector store created successfully.")
