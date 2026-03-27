""" 
microgpt.py — Karpathy's microgpt faithfully ported to Python. 
No PyTorch/TF. Pure Python + math/numpy for Adam arrays only. 
""" 
import math 
import numpy as np 

# ── RNG ─────────────────────────────────────────────────────────────────────── 
_rng = [42] 

def _seed(s): _rng[0] = int(s) 

def _rand(): 
    _rng[0] = (_rng[0] * 1664525 + 1013904223) & 0xFFFFFFFF 
    return _rng[0] / 0xFFFFFFFF 

def _gauss(std): 
    u1 = max(_rand(), 1e-10) 
    return std * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * _rand()) 

def _sample(probs): 
    r, c = np.random.random(), 0.0 
    for i, p in enumerate(probs): 
        c += p 
        if r <= c: return i 
    return len(probs) - 1 

# ── Autograd ────────────────────────────────────────────────────────────────── 
class Value: 
    __slots__ = ('data', 'grad', '_ch', '_lg') 

    def __init__(self, data, ch=(), lg=()): 
        self.data = float(data); self.grad = 0.0 
        self._ch = ch; self._lg = lg 

    def __add__(self, o): 
        o = o if isinstance(o, Value) else Value(o) 
        return Value(self.data + o.data, (self, o), (1.0, 1.0)) 
    __radd__ = __add__ 

    def __mul__(self, o): 
        o = o if isinstance(o, Value) else Value(o) 
        return Value(self.data * o.data, (self, o), (o.data, self.data)) 
    __rmul__ = __mul__ 

    def __neg__(self): return self * -1 
    def __sub__(self, o): return self + (o if isinstance(o, Value) else Value(o)).__neg__() 
    def __truediv__(self, o): 
        o = o if isinstance(o, Value) else Value(o) 
        return self * o.pow(-1) 

    def pow(self, n): 
        return Value(self.data ** n, (self,), (n * self.data ** (n - 1),)) 
    def log(self): 
        d = max(self.data, 1e-10) 
        return Value(math.log(d), (self,), (1.0 / d,)) 
    def exp(self): 
        v = math.exp(self.data); return Value(v, (self,), (v,)) 
    def relu(self): 
        return Value(max(0.0, self.data), (self,), (1.0 if self.data > 0 else 0.0,)) 

    def backward(self): 
        topo, seen = [], set() 
        def build(v): 
            if id(v) not in seen: 
                seen.add(id(v)) 
                for c in v._ch: build(c) 
                topo.append(v) 
        build(self); self.grad = 1.0 
        for v in reversed(topo): 
            for c, g in zip(v._ch, v._lg): 
                c.grad += g * v.grad 

# ── Helpers ─────────────────────────────────────────────────────────────────── 
def linear(x, w): 
    return [sum((w[i][j] * x[j] for j in range(len(x))), Value(0)) for i in range(len(w))] 

def softmax(logits): 
    mx = max(v.data for v in logits) 
    exps = [(v - mx).exp() for v in logits] 
    tot = sum(exps, Value(0)) 
    return [e / tot for e in exps] 

def rmsnorm(x): 
    ms = sum((xi * xi for xi in x), Value(0)) / len(x) 
    s = (ms + 1e-5).pow(-0.5) 
    return [xi * s for xi in x] 

# ── Model ───────────────────────────────────────────────────────────────────── 
class MicroGPT: 
    def __init__(self): 
        self.sd = {}; self.params = [] 
        self.adam_m = self.adam_v = None 
        self.cfg = {}; self.uchars = []; self.BOS = 0 
        self.vocab_size = 0; self.docs = [] 

    def _mat(self, r, c, std=0.08): 
        return [[Value(_gauss(std)) for _ in range(c)] for _ in range(r)] 

    # ── Dataset ─────────────────────────────────────────────────────────────── 
    def load_dataset(self, text, sep='\n'): 
        self.docs = [l.strip() for l in text.strip().split(sep) if l.strip()] 
        for i in range(len(self.docs) - 1, 0, -1): 
            j = int(_rand() * (i + 1)) 
            self.docs[i], self.docs[j] = self.docs[j], self.docs[i] 
        self.uchars = sorted(set(ch for d in self.docs for ch in d)) 
        self.BOS = len(self.uchars) 
        self.vocab_size = self.BOS + 1 
        return dict(vocab_size=self.vocab_size, num_docs=len(self.docs), 
                chars=self.uchars, sample_docs=self.docs[:10]) 

    # ── Init ────────────────────────────────────────────────────────────────── 
    def init_model(self, cfg): 
        self.cfg = dict(cfg); _seed(cfg.get('seed', 42)) 
        E, L, B = cfg['n_embd'], cfg['n_layer'], cfg['block_size'] 
        V = self.vocab_size 
        sd = {'wte': self._mat(V, E), 'wpe': self._mat(B, E), 'lm_head': self._mat(V, E)} 
        for i in range(L): 
            for k in ('attn_wq','attn_wk','attn_wv','attn_wo'): 
                sd[f'layer{i}.{k}'] = self._mat(E, E) 
            sd[f'layer{i}.mlp_fc1'] = self._mat(4*E, E) 
            sd[f'layer{i}.mlp_fc2'] = self._mat(E, 4*E) 
        self.sd = sd 
        self.params = [p for m in sd.values() for row in m for p in row] 
        self.adam_m = np.zeros(len(self.params)) 
        self.adam_v = np.zeros(len(self.params)) 
        return len(self.params) 

    # ── Forward ─────────────────────────────────────────────────────────────── 
    def _forward(self, tok, pos, keys, vals): 
        E, H, L, B = self.cfg['n_embd'], self.cfg['n_head'], self.cfg['n_layer'], self.cfg['block_size'] 
        hd = E // H; sd = self.sd 
        x = [sd['wte'][tok][i] + sd['wpe'][pos % B][i] for i in range(E)] 
        x = rmsnorm(x) 
        for li in range(L): 
            xr = x; x = rmsnorm(x) 
            q = linear(x, sd[f'layer{li}.attn_wq']) 
            k = linear(x, sd[f'layer{li}.attn_wk']) 
            v = linear(x, sd[f'layer{li}.attn_wv']) 
            keys[li].append(k); vals[li].append(v) 
            xa = [] 
            for h in range(H): 
                hs = h * hd 
                qh = q[hs:hs+hd] 
                kh = [ki[hs:hs+hd] for ki in keys[li]] 
                vh = [vi[hs:hs+hd] for vi in vals[li]] 
                al = [sum((qh[j]*kt[j] for j in range(hd)), Value(0)) / math.sqrt(hd) for kt in kh] 
                aw = softmax(al) 
                for j in range(hd): 
                    xa.append(sum((aw[t]*vh[t][j] for t in range(len(vh))), Value(0))) 
            x = linear(xa, sd[f'layer{li}.attn_wo']) 
            x = [a+b for a,b in zip(x, xr)] 
            xr2 = x; x = rmsnorm(x) 
            x = linear(x, sd[f'layer{li}.mlp_fc1']) 
            x = [xi.relu() for xi in x] 
            x = linear(x, sd[f'layer{li}.mlp_fc2']) 
            x = [a+b for a,b in zip(x, xr2)] 
        return linear(x, sd['lm_head']) 

    # ── Generate ────────────────────────────────────────────────────────────── 
    def generate(self, prompt='', temperature=0.5, max_len=None): 
        if max_len is None: max_len = self.cfg['block_size'] 
        L = self.cfg['n_layer'] 
        keys, vals = [[] for _ in range(L)], [[] for _ in range(L)] 

        def _next(tok, pos): 
            logits = self._forward(tok, pos, keys, vals) 
            probs = softmax([l / temperature for l in logits]) 
            return _sample([p.data for p in probs]) 

        if prompt: 
            tok = self.BOS 
            logits = self._forward(tok, 0, keys, vals) 
            for i, ch in enumerate(prompt): 
                if ch in self.uchars: 
                    tok = self.uchars.index(ch) 
                    logits = self._forward(tok, i+1, keys, vals) 
                    probs = softmax([l / temperature for l in logits]) 
                    tok = _sample([p.data for p in probs]) 
                if tok == self.BOS: return prompt or '(empty)' 
            out = [prompt, self.uchars[tok]] 
            for pos in range(len(prompt)+2, max_len): 
                tok = _next(tok, pos-1) 
                if tok == self.BOS: break 
                out.append(self.uchars[tok]) 
            return ''.join(out) 
        else: 
            tok = self.BOS; out = [] 
            for pos in range(max_len): 
                tok = _next(tok, pos) 
                if tok == self.BOS: break 
                out.append(self.uchars[tok]) 
            return ''.join(out) or '(empty)' 

    # ── Train step ──────────────────────────────────────────────────────────── 
    def train_step(self, step, total_steps): 
        B, L = self.cfg['block_size'], self.cfg['n_layer'] 
        lr, b1, b2, eps = 0.01, 0.85, 0.99, 1e-8 
        doc = self.docs[step % len(self.docs)] 
        tokens = [self.BOS] + [self.uchars.index(c) for c in doc if c in self.uchars] + [self.BOS] 
        n = min(B, len(tokens) - 1) 
        keys, vals = [[] for _ in range(L)], [[] for _ in range(L)] 
        losses = [] 
        for pos in range(n): 
            logits = self._forward(tokens[pos], pos, keys, vals) 
            probs = softmax(logits) 
            losses.append(-probs[tokens[pos+1]].log()) 
        loss = sum(losses, Value(0)) / n 
        loss.backward() 
        lr_t = lr * (1 - step / total_steps) 
        for i, p in enumerate(self.params): 
            self.adam_m[i] = b1*self.adam_m[i] + (1-b1)*p.grad 
            self.adam_v[i] = b2*self.adam_v[i] + (1-b2)*p.grad**2 
            mh = self.adam_m[i] / (1 - b1**(step+1)) 
            vh = self.adam_v[i] / (1 - b2**(step+1)) 
            p.data -= lr_t * mh / (math.sqrt(vh) + eps) 
            p.grad = 0.0 
        return loss.data, lr_t, doc
