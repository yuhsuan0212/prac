import jax
import jax.numpy as jnp
import flax.nnx as nnx


def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))


dtype = jnp.bfloat16
param_dtype = jnp.float32


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.layer_norm1 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), None),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), None),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), None),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), None),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(rate=dropout_rate)  # Added dropout layer after MHA
        self.layer_norm2 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), None),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), None),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=embed_dim,
            out_features=ff_dim,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), None),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), None),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=ff_dim,
            out_features=embed_dim,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), None),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), None),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=dropout_rate)

    def __call__(self, inputs, training: bool = False):
        input_shape = inputs.shape
        bs, seq_len, emb_sz = input_shape

        attention_output = self.mha(
            inputs_q=self.layer_norm1(inputs),
            mask=causal_attention_mask(seq_len),
            decode=False,
        )
        x = inputs + self.dropout1(attention_output, deterministic=not training)

        # MLP
        mlp_output = self.linear1(self.layer_norm2(x))
        mlp_output = nnx.gelu(mlp_output)
        mlp_output = self.linear2(mlp_output)
        mlp_output = self.dropout2(mlp_output, deterministic=not training)

        return x + mlp_output


class TokenAndPositionEmbedding(nnx.Module):

    def __init__(self, seqlen: int, vocab_size: int, embed_dim: int, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(
            num_embeddings=vocab_size,
            features=embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.pos_emb = nnx.Embed(
            num_embeddings=seqlen,
            features=embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        position_embedding = self.pos_emb(positions)
        token_embedding = self.token_emb(x)
        return self.token_emb, token_embedding + position_embedding


class GPT2(nnx.Module):
    def __init__(
        self,
        seqlen: int,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        rate: float,
        feed_forward_dim: int,
        num_transformer_blocks: int,
        rngs: nnx.Rngs,
    ):
        self.embedding_layer = TokenAndPositionEmbedding(
            seqlen, vocab_size, embed_dim, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=rate)

        self.transformer_blocks = nnx.List(
            [
                TransformerBlock(
                    embed_dim, num_heads, feed_forward_dim, rate, rngs=rngs
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        self.layer_norm = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), None),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), None),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, inputs, training: bool = False):
        token_embedding, x = self.embedding_layer(inputs)
        x = self.dropout(x, deterministic=not training)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        x = self.layer_norm(x)
        # Weights tying
        outputs = token_embedding.attend(x)
        return outputs

    @nnx.jit
    def sample_from(self, logits):
        logits, indices = jax.lax.top_k(logits, k=top_k)
        logits = nnx.softmax(logits)
        return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

    @nnx.jit
    def generate_step(self, padded_tokens, sample_index):
        logits = self(padded_tokens)
        next_token = self.sample_from(logits[0][sample_index])
        return next_token

    def generate_text(self, max_tokens, start_tokens):
        generated = []
        print(tokenizer.decode(start_tokens), flush=True, end="")
        for i in range(max_tokens):
            sample_index = len(start_tokens) + len(generated) - 1
            # TODO: use attention masking for better efficiency
            padded_tokens = jnp.array(
                (
                    start_tokens
                    + generated
                    + [0] * (seqlen - len(start_tokens) - len(generated))
                )
            )[None, :]
            next_token = int(self.generate_step(padded_tokens, sample_index))
            if (
                next_token
                == tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[
                    0
                ]
            ):
                break
            generated.append(next_token)
            # decode and print next_token
            print(tokenizer.decode([next_token]), flush=True, end="")
        return tokenizer.decode(start_tokens + generated)


def create_model(rngs):
    return GPT2(
        seqlen,
        vocab_size,
        embed_dim,
        num_heads,
        dropout_rate,
        feed_forward_dim,
        num_transformer_blocks,
        rngs=rngs,
    )
