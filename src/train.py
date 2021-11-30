import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------Transformer--------------------------------

# Transformer Parameters
d_model = 512  # Embedding Size（token embedding和position编码的维度）
d_ff = 2048  # FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
d_k = d_v = 64  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 6  # number of Encoder of Decoder Layer（Block的个数）
n_heads = 8  # number of heads in Multi-Head Attention（有几套头）


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），
    例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is <pad> token
    # [batch_size, 1, len_k], True is masked
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # [batch_size, len_q, len_k] (batch_size个len_q * len_k的矩阵)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    """建议打印出来看看是什么的输出（一目了然）
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# ==========================================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1的元素）
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)
        # 对scores的最后一个维度(v)做softmax，得到attn注意力稀疏矩阵
        attn = nn.Softmax(dim=-1)(scores)
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        # context：上下文向量，attn：注意力稀疏矩阵
        return context, attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.projection = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K 和 V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        # 再做一个projection
        output = self.projection(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn



# 残差网络加正则化
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )# 每个位置用同一个全连接网络
        self.layer_norm = nn.LayerNorm(d_model).to(device)

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        # return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]
        return self.layer_norm(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        # 这里的Q,K,V全是Decoder自己的输入
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn  # dec_self_attn, dec_enc_attn这两个是为了可视化的


class Encoder(nn.Module):
    def __init__(self, src_vocab_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(d_model)  # Transformer中位置编码固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # 获取Encoder输入序列的pad mask矩阵
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        enc_self_attns = []
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(
            device)  # [batch_size, tgt_len, d_model]
        # Decoder输入序列的pad mask矩阵
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)  # [batch_size, tgt_len, tgt_len]
        # Masked Self_Attention：当前时刻是看不到后面时刻的信息的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
            device)  # [batch_size, tgt_len, tgt_len]

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        # torch.gt(input, other, *, out=None) → Tensor
        # Computes \text{input} > \text{other}input>other element-wise.
        # return true or false tensor
        # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device)

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵
        # (因为enc是处理K,V的，
        # 求Attention时是用v1,v2,..vm去加权的，
        # 要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        # [batc_size, tgt_len, src_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model],
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len],
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size).to(device)
        self.decoder = Decoder(tgt_vocab_size).to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        """Transformers的输入：两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model]
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


# --------------------------------------end----------------------------------------------------


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


def make_data(src_sentences, tgt_sentences, n_step):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(src_sentences)):
        src_tokens = src_sentences[i].strip().split(" ")
        tgt_tokens1 = tgt_sentences[i].strip().split(" ")
        tgt_tokens2 = tgt_sentences[i].strip().split(" ")

        # 为长度不足n_step的句子补上<pad>符号
        if len(src_tokens) <= n_step:
            src_tokens = ['<pad>'] * (n_step + 1 - len(src_tokens)) + src_tokens
        if len(tgt_tokens1) <= n_step:
            tgt_tokens1 = ['<pad>'] * (n_step + 1 - len(tgt_tokens1)) + tgt_tokens1
            tgt_tokens2 = ['<pad>'] * (n_step + 1 - len(tgt_tokens2)) + tgt_tokens2

        # 为目标语言句子添加<eos>,<sos>符号
        if tgt_tokens1[0] != '<sos>':
            tgt_tokens1 = ['<sos>'] + tgt_tokens1
        if tgt_tokens2[-1] != '<eos>':
            tgt_tokens2.append('<eos>')

        enc_input = [[src_word2number[n] if n in src_word2number else src_word2number['<unk_word>'] for n in src_tokens]]
        dec_input = [[tgt_word2number[n] if n in tgt_word2number else tgt_word2number['<unk_word>'] for n in tgt_tokens1]]
        dec_output = [[tgt_word2number[n] if n in tgt_word2number else tgt_word2number['<unk_word>'] for n in tgt_tokens2]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


def get_dataset(src_path, tgt_path):
    text_de = open(src_path, 'r', encoding='utf-8')
    text_en = open(tgt_path, 'r', encoding='utf-8')
    src_word_list = set()
    tgt_word_list = set()
    src_sen_list = []
    tgt_sen_list = []
    max_seq_len = 0
    idx = 0
    idxs = []
    for line in text_de:
        words = line.strip().split(" ")  # 去空格按空格拆分
        src_word_list = src_word_list.union(set(words))
        if len(words) < 50:#选出长度小于50的句子
            max_seq_len = max(len(words), max_seq_len)
            idxs.append(idx)
            src_sen_list.append(line)
        idx = idx + 1

    idx = 0
    for line in text_en:
        words = line.strip().split(" ")
        tgt_word_list = tgt_word_list.union(set(words))
        if idx in idxs:
            max_seq_len = max(len(words), max_seq_len)
            tgt_sen_list.append(line)
        idx = idx + 1

    src_word_list = list(sorted(src_word_list))
    tgt_word_list = list(sorted(tgt_word_list))

    return max_seq_len, src_sen_list, tgt_sen_list, src_word_list, tgt_word_list


def get_dict(src_word_list, tgt_word_list):
    # 获取词到数字字典
    src_word2number = {w: i + 4 for i, w in enumerate(src_word_list)}
    tgt_word2number = {w: i + 4 for i, w in enumerate(tgt_word_list)}
    # 获取数字到词字典
    src_number2word = {i + 4: w for i, w in enumerate(src_word_list)}
    tgt_number2word = {i + 4: w for i, w in enumerate(tgt_word_list)}
    # 添加特殊符号到字典：
    # <pad>补齐句子用的填充符号
    # <unk_word>未知单词
    # <sos>开始符号
    # <eos>结束符号
    src_word2number["<pad>"] = 0
    src_word2number["<unk_word>"] = 1
    src_word2number["<sos>"] = 2
    src_word2number["<eos>"] = 3
    src_number2word[0] = "<pad>"
    src_number2word[1] = "<unk_word>"
    src_number2word[2] = "<sos>"
    src_number2word[3] = "<eos>"
    tgt_word2number["<pad>"] = 0
    tgt_word2number["<unk_word>"] = 1
    tgt_word2number["<sos>"] = 2
    tgt_word2number["<eos>"] = 3
    tgt_number2word[0] = "<pad>"
    tgt_number2word[1] = "<unk_word>"
    tgt_number2word[2] = "<sos>"
    tgt_number2word[3] = "<eos>"

    return src_word2number, src_number2word, tgt_word2number, tgt_number2word


def greedy_decoder(model, enc_input, start_symbol):
    """贪心编码
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        # 推测阶段：dec_input序列会一点点变长（每次添加一个新推测出来的单词）
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 取出当前预测的单词，只取最新预测的单词拼接到输入序列中
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_word2number["<eos>"]:
            terminal = True

    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict



def train(model):
    model.train()
    train_loss = []
    train_ppl = []
    for epoch in range(epochs):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            # 梯度清零
            optimizer.zero_grad()
            # enc_inputs: [batch_size, src_len]
            # dec_inputs: [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)

            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

            # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            train_loss.append(loss.item())
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 计算ppl困惑度
            ppl = math.exp(loss.item())
            train_ppl.append(ppl)
        # 打印结果
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(np.mean(train_loss)), 'ppl:', '{:.6}'.format(np.mean(train_ppl)))

        # 训练一代，验证一次
        model.eval()
        val_loss = []
        val_ppl = []
        with torch.no_grad():
            for v_enc_inputs, v_dec_inputs, v_dec_outputs in v_loader:
                v_enc_inputs, v_dec_inputs, v_dec_outputs = v_enc_inputs.to(device), v_dec_inputs.to(device), v_dec_outputs.to(device)
                v_outputs, v_enc_self_attns, v_dec_self_attns, v_dec_enc_attns = model(v_enc_inputs, v_dec_inputs)

                # 计算loss
                loss = criterion(v_outputs, v_dec_outputs.view(-1))
                val_loss.append(loss.item())
                # 计算ppl困惑度
                ppl = math.exp(loss.item())
                val_ppl.append(ppl)
        # 打印结果
        print('Val:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(np.mean(val_loss)), 'ppl:', '{:.6}'.format(np.mean(val_ppl)))

        # 每save_interval代保存一次模型
        if (epoch + 1) % save_interval == 0:
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model, f'models/epoch_{epoch + 1}.ckpt')


def test(model):
    test_loss = []
    test_ppl = []
    with torch.no_grad():
        for t_enc_inputs, t_dec_inputs, t_dec_outputs in t_loader:
            t_enc_inputs, t_dec_inputs, t_dec_outputs = t_enc_inputs.to(device), t_dec_inputs.to(device), t_dec_outputs.to(device)
            t_outputs, t_enc_self_attns, t_dec_self_attns, t_dec_enc_attns = model(t_enc_inputs, t_dec_inputs)

            # 计算loss
            loss = criterion(t_outputs, t_dec_outputs.view(-1))
            test_loss.append(loss.item())
            # 计算ppl困惑度
            ppl = math.exp(loss.item())
            test_ppl.append(ppl)
    # 打印结果
    print('Test: loss =', '{:.6f}'.format(np.mean(test_loss)), 'ppl:', '{:.6}'.format(np.mean(test_ppl)))
    #-----------------------------------------推测--------------------------------------------
    t_enc_inputs, _, _ = next(iter(t_loader))
    for i in range(len(t_enc_inputs)):
        greedy_dec_predict = greedy_decoder(model, t_enc_inputs[i].view(1, -1).to(device), start_symbol=tgt_word2number["<sos>"])
        print(t_enc_inputs[i], '->', greedy_dec_predict.squeeze())
        print([src_number2word[t.item()] for t in t_enc_inputs[i]], '->', [tgt_number2word[n.item()] for n in greedy_dec_predict.squeeze()])


if __name__ == "__main__":
    # 参数
    epochs = 1
    learning_rate = 1e-3
    batch_size = 1
    save_interval = 10
    momentum = 0.99
    # 数据集路径
    src_train_path = '../DataSet/train.de'
    tgt_train_path = '../DataSet/train.en'
    src_validate_path = '../DataSet/valid.de'
    tgt_validate_path = '../DataSet/valid.en'
    src_test_path = '../DataSet/test.de'
    tgt_test_path = '../DataSet/test.de'
    # 生成训练数据
    max_seq_len, src_sen_list, tgt_sen_list, src_word_list, tgt_word_list = get_dataset(src_train_path, tgt_train_path)
    src_word2number, src_number2word, tgt_word2number, tgt_number2word = get_dict(src_word_list, tgt_word_list)
    # 生成验证数据
    v_max_seq_len, v_src_sen_list, v_tgt_sen_list, v_src_word_list, v_tgt_word_list = get_dataset(src_validate_path, tgt_validate_path)
    # 生成测试数据
    t_max_seq_len, t_src_sen_list, t_tgt_sen_list, t_src_word_list, t_tgt_word_list = get_dataset(src_test_path, tgt_test_path)
    max_seq_len = max(max_seq_len, v_max_seq_len, t_max_seq_len)

    # 把单词序列转换为数字序列
    enc_inputs, dec_inputs, dec_outputs = make_data(src_sen_list, tgt_sen_list, max_seq_len)
    v_enc_inputs, v_dec_inputs, v_dec_outputs = make_data(v_src_sen_list, v_tgt_sen_list, max_seq_len)
    t_enc_inputs, t_dec_inputs, t_dec_outputs = make_data(t_src_sen_list, t_tgt_sen_list, max_seq_len)

    # 把输入按batch_size封装到DataLoader中
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size, True)
    v_loader = Data.DataLoader(MyDataSet(v_enc_inputs, v_dec_inputs, v_dec_outputs), batch_size, True)
    t_loader = Data.DataLoader(MyDataSet(t_enc_inputs, t_dec_inputs, t_dec_outputs), batch_size, True)

    src_vocab_size = len(src_number2word)
    tgt_vocab_size = len(tgt_number2word)

    # 定义模型
    model = Transformer(src_vocab_size, tgt_vocab_size).to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # 定义SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(model.parameters(),lr=learning_rate, betas=(0.9, 0.98))
    # 训练
    train(model)
    # 测试
    test(model)