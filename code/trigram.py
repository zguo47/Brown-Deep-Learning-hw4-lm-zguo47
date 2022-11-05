import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyTrigram(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=100, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        ## TODO: Finish off the method as necessary

        self.embedding_table = tf.keras.layers.Embedding(self.vocab_size, self.embed_size)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.vocab_size, activation='softmax', dtype = 'float32')
        ])


    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)
        :return: logits: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        ## TODO: Implement the method as necessary

        # X_trigram = np.vstack([inputs[0:-2], inputs[1:-1]]).T

        # y_trigram = inputs[2:]

        # tf_embedding_table = tf.Variable(tf.random.normal([inputs.shape[0], inputs.shape[1]], stddev=0.01, dtype=tf.float32))
        # tf_embedding_vectors = tf.nn.embedding_lookup(tf_embedding_table, X_trigram)
        # tf_embedding_vectors = tf.reshape(tf_embedding_vectors, output_shape)

        embedding_vector0 = self.embedding_table(inputs[:, 0])
        embedding_vector1 = self.embedding_table(inputs[:, 1])
        embedding_input = tf.concat([embedding_vector0, embedding_vector1], axis = 1)
        outputs = self.model(embedding_input)

        return outputs

    def generate_sentence(self, word1, word2, length, vocab):
        """
        Given initial 2 words, print out predicted sentence of targeted length.

        :param word1: string, first word
        :param word2: string, second word
        :param length: int, desired sentence length
        :param vocab: dictionary, word to id mapping
        :param model: trained trigram model

        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        output_string = np.zeros((1, length), dtype=np.int)
        output_string[:, :2] = vocab[word1], vocab[word2]

        for end in range(2, length):
            start = end - 2
            output_string[:, end] = np.argmax(self(output_string[:, start:end]), axis=1)
        text = [reverse_vocab[i] for i in list(output_string[0])]

        print(" ".join(text))


#########################################################################################

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MyTrigram(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy()

    class Perplexity(tf.keras.losses.SparseCategoricalCrossentropy):
        def __init__(self, *args, name="perplexity", **kwargs):
            super().__init__(*args, name="perplexity", **kwargs)
        
        def __call__(self, *args, **kwds):
            return tf.exp(tf.reduce_mean(super().__call__(*args, **kwds)))


    acc_metric  = Perplexity()

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001), 
        loss=loss_metric, 
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model = model,
        epochs = 1,
        batch_size = 100,
    )


#########################################################################################

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: You might be able to find this somewhere...
    data_path = "../data"
    train_id, test_id, word_to_token_dict = get_data(f"{data_path}/train.txt", f"{data_path}/test.txt")

    vocab = word_to_token_dict

    X0, Y0  = np.array(train_id[:-1]), np.array(train_id[2:])
    X1, Y1  = np.array(test_id[:-1]), np.array(test_id[2:])
    X0 = np.column_stack((X0[:-1], X0[1:]))
    X1 = np.column_stack((X1[:-1], X1[1:]))

    # TODO: Get your model that you'd like to use
    args = get_text_model(vocab)

    # TODO: Implement get_text_model to return the model that you want to use. 
    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    words = 'speak to this brown deep learning student'.split()
    for word1, word2 in zip(words[:-1], words[1:]):
        if word1 not in vocab: print(f"{word1} not in vocabulary")
        if word2 not in vocab: print(f"{word2} not in vocabulary")
        else: args.model.generate_sentence(word1, word2, 20, vocab)

if __name__ == '__main__':
    main()
