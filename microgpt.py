import math
import random
from typing import List

import numpy as np


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data + 1e-8), (self,), "log")

        def _backward():
            self.grad += (1.0 / (self.data + 1e-8)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def linear(x: List[Value], w: List[List[Value]]) -> List[Value]:
    return [sum((wi * xi for wi, xi in zip(wrow, x)), Value(0.0)) for wrow in w]


def softmax(logits: List[Value]) -> List[Value]:
    max_logit = max(l.data for l in logits)
    probs = [(l - max_logit).exp() for l in logits]
    denom = sum(probs, Value(0.0))
    return [p / denom for p in probs]


def rmsnorm(x: List[Value]) -> List[Value]:
    ss = sum((xi * xi for xi in x), Value(0.0)) / len(x)
    scale = (ss + 1e-5) ** -0.5
    return [xi * scale for xi in x]


class MicroGPT:
    def __init__(self, seed: int = 42):
        self.rng_state = seed
        self.docs = []
        self.stoi = {}
        self.itos = {}
        self.BOS = 0
        self.params = []
        self.cfg = {}
        self.initialized = False

    def _seed(self, s: int):
        self.rng_state = s

    def _rand(self) -> float:
        self.rng_state = (1664525 * self.rng_state + 1013904223) % (2**32)
        return self.rng_state / (2**32)

    def _gauss(self, mu=0.0, sigma=1.0):
        u1 = max(self._rand(), 1e-8)
        u2 = self._rand()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z0

    def _sample(self, probs: List[float]):
        r = self._rand()
        cdf = 0.0
        for i, p in enumerate(probs):
            cdf += p
            if r <= cdf:
                return i
        return len(probs) - 1

    def load_dataset(self, text: str):
        docs = [d.strip() for d in text.splitlines() if d.strip()]
        random.Random(1337).shuffle(docs)
        if not docs:
            raise ValueError("Dataset is empty.")
        uchars = sorted(set("".join(docs)))
        self.stoi = {ch: i for i, ch in enumerate(uchars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.BOS = len(uchars)
        self.stoi["<BOS>"] = self.BOS
        self.itos[self.BOS] = "<BOS>"
        self.docs = docs
        return {
            "num_docs": len(docs),
            "vocab_size": len(uchars),
            "avg_len": float(np.mean([len(d) for d in docs])),
            "max_len": int(np.max([len(d) for d in docs])),
            "samples": docs[:6],
        }

    def _rand_matrix(self, rows, cols, scale=0.02):
        return [[Value(self._gauss(0, scale)) for _ in range(cols)] for _ in range(rows)]

    def init_model(self, cfg):
        self.cfg = cfg
        V = len(self.stoi)
        C = cfg["n_embd"]
        T = cfg["block_size"]
        L = cfg["n_layer"]

        self.wte = self._rand_matrix(V, C)
        self.wpe = self._rand_matrix(T, C)
        self.lm_head = self._rand_matrix(V, C)

        self.attn_wq = [self._rand_matrix(C, C) for _ in range(L)]
        self.attn_wk = [self._rand_matrix(C, C) for _ in range(L)]
        self.attn_wv = [self._rand_matrix(C, C) for _ in range(L)]
        self.attn_wo = [self._rand_matrix(C, C) for _ in range(L)]

        self.mlp_fc1 = [self._rand_matrix(4 * C, C) for _ in range(L)]
        self.mlp_fc2 = [self._rand_matrix(C, 4 * C) for _ in range(L)]

        mats = [self.wte, self.wpe, self.lm_head]
        for l in range(L):
            mats += [
                self.attn_wq[l],
                self.attn_wk[l],
                self.attn_wv[l],
                self.attn_wo[l],
                self.mlp_fc1[l],
                self.mlp_fc2[l],
            ]

        self.params = [p for m in mats for row in m for p in row]
        n = len(self.params)
        self.m = np.zeros(n, dtype=np.float64)
        self.v = np.zeros(n, dtype=np.float64)
        self.initialized = True

    def _forward(self, tok, pos, keys=None, vals=None):
        C = self.cfg["n_embd"]
        L = self.cfg["n_layer"]
        n_head = self.cfg["n_head"]
        head_dim = C // n_head
        x = [self.wte[tok][i] + self.wpe[pos][i] for i in range(C)]

        for l in range(L):
            xn = rmsnorm(x)
            q = linear(xn, self.attn_wq[l])
            k = linear(xn, self.attn_wk[l])
            v = linear(xn, self.attn_wv[l])

            if keys is None:
                k_cache = [[] for _ in range(L)]
                v_cache = [[] for _ in range(L)]
            else:
                k_cache, v_cache = keys, vals
            k_cache[l].append(k)
            v_cache[l].append(v)

            head_out = []
            for h in range(n_head):
                qs = q[h * head_dim : (h + 1) * head_dim]
                scores = []
                for t in range(len(k_cache[l])):
                    ks = k_cache[l][t][h * head_dim : (h + 1) * head_dim]
                    dot = sum((a * b for a, b in zip(qs, ks)), Value(0.0))
                    scores.append(dot * (head_dim**-0.5))
                probs = softmax(scores)
                out = [Value(0.0) for _ in range(head_dim)]
                for t, p in enumerate(probs):
                    vs = v_cache[l][t][h * head_dim : (h + 1) * head_dim]
                    out = [oi + p * vi for oi, vi in zip(out, vs)]
                head_out.extend(out)

            attn = linear(head_out, self.attn_wo[l])
            x = [xi + ai for xi, ai in zip(x, attn)]

            xn2 = rmsnorm(x)
            h1 = [u.relu() for u in linear(xn2, self.mlp_fc1[l])]
            h2 = linear(h1, self.mlp_fc2[l])
            x = [xi + hi for xi, hi in zip(x, h2)]

        xn = rmsnorm(x)
        logits = linear(xn, self.lm_head)
        return logits, k_cache, v_cache

    def _encode_doc(self, doc):
        return [self.BOS] + [self.stoi[ch] for ch in doc if ch in self.stoi]

    def train_step(self, step, total_steps):
        if not self.initialized:
            raise RuntimeError("Model not initialized")

        for p in self.params:
            p.grad = 0.0

        doc = self.docs[step % len(self.docs)]
        toks = self._encode_doc(doc)
        if len(toks) < 2:
            return 0.0, doc

        block = self.cfg["block_size"]
        toks = toks[: block + 1]
        xseq = toks[:-1]
        yseq = toks[1:]

        loss = Value(0.0)
        keys = [[] for _ in range(self.cfg["n_layer"])]
        vals = [[] for _ in range(self.cfg["n_layer"])]

        for t, (xt, yt) in enumerate(zip(xseq, yseq)):
            logits, keys, vals = self._forward(xt, t, keys, vals)
            probs = softmax(logits)
            loss = loss + (probs[yt].log() * -1.0)

        loss = loss / max(1, len(xseq))
        loss.backward()

        lr = 0.01 * max(0.0, 1.0 - step / max(1, total_steps))
        b1, b2 = 0.85, 0.99
        eps = 1e-8
        t = step + 1

        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g * g)
            m_hat = self.m[i] / (1 - (b1**t))
            v_hat = self.v[i] / (1 - (b2**t))
            p.data -= lr * m_hat / (math.sqrt(v_hat) + eps)

        return float(loss.data), doc

    def generate(self, prompt="", temperature=1.0, max_len=30):
        if not self.initialized:
            return ""
        toks = [self.BOS] + [self.stoi[c] for c in prompt if c in self.stoi]
        keys = [[] for _ in range(self.cfg["n_layer"])]
        vals = [[] for _ in range(self.cfg["n_layer"])]

        for t, tok in enumerate(toks):
            logits, keys, vals = self._forward(tok, min(t, self.cfg["block_size"] - 1), keys, vals)

        out = []
        cur_logits = logits
        for i in range(max_len):
            scaled = [Value(l.data / max(temperature, 1e-4)) for l in cur_logits]
            probs = softmax(scaled)
            pvals = [p.data for p in probs]
            nxt = self._sample(pvals)
            if nxt == self.BOS:
                break
            out.append(self.itos.get(nxt, ""))
            cur_logits, keys, vals = self._forward(
                nxt,
                min(len(toks) + i, self.cfg["block_size"] - 1),
                keys,
                vals,
            )

        return "".join(out)
