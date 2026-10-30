"""
Лабораторна робота №4. Приховування даних у частотній області зображень (ДКП).

  Завдання 1 - пряме/зворотне ДКП, частотна чутливість зору
  Завдання 2 - метод Коха-Жао, коефіцієнти (3,1) та (1,3)
  Завдання 3 - JPEG-атака, залежності Po(Pr) та RAZ(Pr)
  Завдання 4 - метод Бенгама-Мемона-Ео-Юнг (3 коефіцієнти) + відбракування блоків PL/PH
  Завдання 5 - (додаткове) метод Дж. Фрідріх

Як у методичці: контейнер - BMP24, вбудовування в червоний канал,
повідомлення - txt (біти кожного байта беруться від молодшого, як у D2B).

Запуск:
    python lab4.py --image 1.bmp --msg 1.txt --quality 75 --pr 5
Залежності: numpy, pillow, matplotlib
"""
import argparse
import io
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 8


# ======================================================================
#  ДКП
# ======================================================================
def dct_matrix(n):
    """Ортонормована матриця ДКП-II розміру n x n."""
    u = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    D = np.cos((2 * x + 1) * u * np.pi / (2 * n)) * np.sqrt(2.0 / n)
    D[0, :] = np.sqrt(1.0 / n)
    return D


D8 = dct_matrix(N)


def split_blocks(img):
    h, w = img.shape
    h8, w8 = h - h % N, w - w % N
    a = img[:h8, :w8].astype(float)
    return a.reshape(h8 // N, N, w8 // N, N).swapaxes(1, 2)       # (mm, nn, 8, 8)


def merge_blocks(blocks):
    mm, nn = blocks.shape[:2]
    return blocks.swapaxes(1, 2).reshape(mm * N, nn * N)


def blocks_dct(img):
    return D8 @ split_blocks(img) @ D8.T


def blocks_idct(C):
    return D8.T @ C @ D8


def to_uint8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)        # як рис. 5.9


def put_back(R, new_region):
    """Повернути оброблену область (кратну 8) у повнорозмірний канал."""
    out = R.copy()
    h, w = new_region.shape
    out[:h, :w] = new_region
    return out


# ======================================================================
#  Повідомлення
# ======================================================================
def bytes_to_bits(data: bytes):
    return np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")   # як D2B


def bits_to_bytes(bits):
    bits = np.asarray(bits, np.uint8)
    bits = bits[: len(bits) // 8 * 8]
    return np.packbits(bits, bitorder="little").tobytes()


def bits_to_text(bits):
    return bits_to_bytes(bits).decode("utf-8", errors="replace")


def shown(bits):
    """Текст для виводу: недруковані символи (сміття після атаки) замінюються на '·'."""
    t = bits_to_text(bits)
    return repr("".join(c if (c.isprintable() and c != "\ufffd") else "·" for c in t))


def sgn(x):
    return -1.0 if x < 0 else 1.0


# ======================================================================
#  Метод Коха-Жао (рис. 5.7): (3,1) і (1,3)
# ======================================================================
def kz_embed_block(c, bit, Pr):
    a, b = abs(c[3, 1]), abs(c[1, 3])
    if bit == 1 and not (a - b > Pr):
        c[3, 1] = sgn(c[3, 1]) * (b + Pr)
    elif bit == 0 and not (a - b < -Pr):
        c[1, 3] = sgn(c[1, 3]) * (a + Pr)


def kz_extract_block(c):
    return 1 if abs(c[3, 1]) > abs(c[1, 3]) else 0


# ======================================================================
#  Метод Бенгама-Мемона-Ео-Юнг (рис. 5.18, 5.22): (3,1), (1,3), (3,2)
#   біт 1: |c(3,1)| більший за |c(1,3)| та |c(3,2)| щонайменше на Pr
#   біт 0: |c(1,3)| та |c(3,2)| більші за |c(3,1)| щонайменше на Pr
# ======================================================================
def bmey_embed_block(c, bit, Pr):
    a31, a13, a32 = abs(c[3, 1]), abs(c[1, 3]), abs(c[3, 2])
    if bit == 1:
        t = max(a13, a32) + Pr
        if a31 < t:
            c[3, 1] = sgn(c[3, 1]) * t
    else:
        t = a31 + Pr
        if a13 < t:
            c[1, 3] = sgn(c[1, 3]) * t
        if a32 < t:
            c[3, 2] = sgn(c[3, 2]) * t


def bmey_extract_block(c):
    a31, a13, a32 = abs(c[3, 1]), abs(c[1, 3]), abs(c[3, 2])
    return 1 if (a31 > a13 and a31 > a32) else 0


# --- друге вдосконалення: відбракування блоків (PL, PH) ---------------
# Модифікуються лише (3,1),(1,3),(3,2), тому критерії будуються на коефіцієнтах,
# яких вбудовування не торкається: НЧ (0,1),(1,0),(1,1) та ВЧ (u+v>=10).
HF_SET = [(u, v) for u in range(N) for v in range(N)
          if u + v >= 4 and (u, v) not in ((3, 1), (1, 3), (3, 2))]
HF_THR = 2.0      # "ненульовий" ВЧ-коефіцієнт (з запасом на шум округлення пікселів)


def block_ok(c, PL, PH):
    """Блок придатний, якщо немає різких переходів (НЧ-коефіцієнти <= PL)
    і він не надто однорідний (ненульових ВЧ-коефіцієнтів не менше PH)."""
    lf = max(abs(c[0, 1]), abs(c[1, 0]), abs(c[1, 1]))
    hf = sum(abs(c[u, v]) >= HF_THR for u, v in HF_SET)
    return lf <= PL and hf >= PH


def suitable_mask(C, reject):
    mm, nn = C.shape[:2]
    if reject is None:
        return np.ones((mm, nn), bool)
    PL, PH = reject
    return np.array([[block_ok(C[i, j], PL, PH) for j in range(nn)] for i in range(mm)])


METHODS = {
    "kz": (kz_embed_block, kz_extract_block),
    "bmey": (bmey_embed_block, bmey_extract_block),
}


def capacity(R, reject=None):
    return int(suitable_mask(blocks_dct(R), reject).sum())


def embed(R, bits, Pr, method, reject=None):
    """Вбудувати біти в канал R (2D uint8). Повертає R2 (uint8).

    Блок змінюється й округлюється до цілих пікселів, тому його ВЧ-коефіцієнти трохи
    "пливуть". Якщо після цього блок перестав би пройти відбракування, приймач пропустив
    би його, і всі наступні біти зсунулися б. Тому за відбракування передавач
    перевіряє кожен блок і за потреби вбудовує з дещо більшим Pr (або лишає блок як є)."""
    emb, _ = METHODS[method]
    C = blocks_dct(R)
    mask = suitable_mask(C, reject)
    pos = np.argwhere(mask)                         # блоки в порядку рядків
    if len(bits) > len(pos):
        raise ValueError(f"Повідомлення ({len(bits)} біт) більше за місткість ({len(pos)})")
    pix = split_blocks(R).copy()
    for k, (i, j) in enumerate(pos[: len(bits)]):
        for extra in range(6 if reject else 1):
            c = C[i, j].copy()
            emb(c, int(bits[k]), Pr + extra)
            px = to_uint8(D8.T @ c @ D8)
            if reject is None or block_ok(D8 @ px.astype(float) @ D8.T, *reject):
                pix[i, j] = px
                break
    return put_back(R, to_uint8(merge_blocks(pix)))


def extract(R, nbits, method, reject=None, mask=None):
    _, ext = METHODS[method]
    C = blocks_dct(R)
    if mask is None:
        mask = suitable_mask(C, reject)
    pos = np.argwhere(mask)[:nbits]
    out = np.zeros(nbits, np.uint8)
    for k, (i, j) in enumerate(pos):
        out[k] = ext(C[i, j])
    return out                                       # якщо блоків бракує - решта нулі


# ======================================================================
#  Метрики та атака
# ======================================================================
def raz(a, b):
    return float(np.mean(np.abs(a.astype(float) - b.astype(float))))


def err_prob(m, m1):
    return float(np.mean(np.asarray(m) != np.asarray(m1)))


def jpeg_attack_rgb(rgb, quality):
    """Стискання JPEG та повернення червоного каналу (READ_RED("Stego.jpg"))."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))[:, :, 0]


def with_red(rgb, R2):
    out = rgb.copy()
    out[:, :, 0] = R2
    return out


# ======================================================================
#  Завдання 1: частотна чутливість зору
# ======================================================================
def task1(R, out):
    print("\n=== Завдання 1: ДКП та частотна чутливість зору ===")
    # перевірка: пряме + зворотне ДКП відновлює зображення
    err = np.abs(merge_blocks(blocks_idct(blocks_dct(R))) - R[: R.shape[0] // 8 * 8, : R.shape[1] // 8 * 8]).max()
    print(f"Макс. похибка відновлення ДКП->ОДКП: {err:.2e}")

    A = 24.0
    cases = [("оригінал", None), ("НЧ (0,1)", (0, 1)), ("СЧ (3,1)", (3, 1)), ("ВЧ (7,6)", (7, 6))]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    base = merge_blocks(blocks_idct(blocks_dct(R)))
    for ax, (name, uv) in zip(axs, cases):
        C = blocks_dct(R)
        if uv:
            C[:, :, uv[0], uv[1]] += A
        img = merge_blocks(blocks_idct(C))
        ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
        print(f"  {name:10s} коефіцієнт +{A:.0f} у кожному блоці, середня зміна пікселя: "
              f"{np.mean(np.abs(img - base)):.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "task1_sensitivity.png"), dpi=130)
    plt.close()


# ======================================================================
#  Завдання 2 і 4: порівняння контейнерів
# ======================================================================
def save_compare(R, R2, name, title, out):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    axs[0].imshow(R, cmap="gray", vmin=0, vmax=255); axs[0].set_title("R (порожній контейнер)")
    axs[1].imshow(R2, cmap="gray", vmin=0, vmax=255); axs[1].set_title(f"R2 ({title})")
    axs[2].imshow(np.abs(R2.astype(int) - R.astype(int)), cmap="magma", vmin=0, vmax=30)
    axs[2].set_title("|R2 - R| (шкала 0..30)")
    for a in axs:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out, name), dpi=120)
    plt.close()


def run_method(title, method, rgb, R, bits, Pr, quality, out, reject=None, tag=None):
    print(f"\n=== {title}, Pr={Pr} ===")
    R2 = embed(R, bits, Pr, method, reject)
    stego = with_red(rgb, R2)
    Image.fromarray(stego).save(os.path.join(out, f"Stego_{tag or method}.bmp"))
    # без атаки приймач сам повторює відбракування (маска обчислюється з R2)
    m1 = extract(R2, len(bits), method, reject)
    att = jpeg_attack_rgb(stego, quality)
    print(f"RAZ (спотворення контейнера)       = {raz(R, R2):.3f}")
    print(f"Po без атаки                       = {err_prob(bits, m1):.3f}")
    print(f"Текст без атаки : {shown(m1)}")
    if reject is None:
        m2 = extract(att, len(bits), method)
        print(f"Po після JPEG (якість {quality})        = {err_prob(bits, m2):.3f}")
    else:
        # JPEG знищує ВЧ-коефіцієнти, тому маска, обчислена з атакованого зображення, не збігається
        # з маскою відправника. Дається обидва варіанти.
        m2 = extract(att, len(bits), method, reject)
        mask_k = suitable_mask(blocks_dct(R2), reject)
        m3 = extract(att, len(bits), method, mask=mask_k)
        print(f"Po після JPEG, маска обчислена з атакованого зображення = {err_prob(bits, m2):.3f}")
        print(f"Po після JPEG, маска блоків відома приймачу (ключ)      = {err_prob(bits, m3):.3f}")
        m2 = m3
    print(f"Текст після JPEG: {shown(m2)}")
    save_compare(R, R2, f"compare_{tag or method}.png", title, out)


# ======================================================================
#  Завдання 3: залежності від Pr
# ======================================================================
def task3(rgb, R, bits, quality, reject, out, prs):
    print(f"\n=== Завдання 3: залежності від Pr (JPEG, якість {quality}) ===")
    variants = [("Кох-Жао", "kz", None), ("Бенгам", "bmey", None)]
    if reject is not None:
        variants.append(("Бенгам+відбр.", "bmey", reject))
    res = {v[0]: {"po": [], "raz": []} for v in variants}
    po0_max = {v[0]: 0.0 for v in variants}
    head = f"{'Pr':>4} | " + " | ".join(f"{n:^17}" for n, _, _ in variants)
    print(head)
    print(f"{'':>4} | " + " | ".join(f"{'Po':>8} {'RAZ':>8}" for _ in variants))
    for pr in prs:
        row = []
        for name, m, rj in variants:
            R2 = embed(R, bits, pr, m, rj)
            po0_max[name] = max(po0_max[name], err_prob(bits, extract(R2, len(bits), m, rj)))
            att = jpeg_attack_rgb(with_red(rgb, R2), quality)
            if rj is None:
                po = err_prob(bits, extract(att, len(bits), m))
            else:   # маска блоків відома приймачу (див. примітку в run_method)
                mk = suitable_mask(blocks_dct(R2), rj)
                po = err_prob(bits, extract(att, len(bits), m, mask=mk))
            rz = raz(R, R2)
            res[name]["po"].append(po)
            res[name]["raz"].append(rz)
            row.append(f"{po:8.3f} {rz:8.3f}")
        print(f"{pr:>4} | " + " | ".join(row))

    print("Po ДО атаки (максимум за всіма Pr): " +
          ", ".join(f"{n} = {po0_max[n]:.3f}" for n in po0_max))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for (name, _, _), ls in zip(variants, ["-", "--", ":"]):
        axs[0].plot(prs, res[name]["po"], ls, marker="o", label=name)
        axs[1].plot(prs, res[name]["raz"], ls, marker="o", label=name)
    axs[1].axhline(256 * 0.02, color="gray", lw=1, label="256*0.02 = 5.12")
    axs[0].set(xlabel="Pr", ylabel="Po", title="Ймовірність помилки вилучення (після JPEG)")
    axs[1].set(xlabel="Pr", ylabel="RAZ", title="Середнє спотворення контейнера")
    for a in axs:
        a.grid(alpha=.3)
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "task3_dependencies.png"), dpi=130)
    plt.close()


# ======================================================================
#  Завдання 5: метод Дж. Фрідріх
#   - НЧ-частина: біти кодуються індексною функцією ind(t) на геометричній прогресії (3.8-3.9)
#   - СЧ-частина: додаткове повідомлення з розширенням спектра (3.12)
#   - детектування: зважена кореляція K (3.10) та K' з пошуком масштабу d (3.11)
#
# Нормування (3.7): G = 1024*(C-mu)/(sqrt(X*Y)*sigma) = 1024*(C-mu)/||C-mu||, тому
# вектор ДКП-коефіцієнтів, який бачить приймач, ЗАВЖДИ має енергію 1024^2.
# Вбудовування змінює енергію, а отже й масштаб, який використає приймач, тому
# передавач підбирає НЧ-значення ітераційно (див. fr_embed), щоб після нормування
# на приймальній стороні коефіцієнти потрапляли в потрібні інтервали.
# ======================================================================
FR_E = 1024.0


def fr_dct(R):
    H, W = R.shape
    mu = R.mean()
    Dy, Dx = dct_matrix(H), dct_matrix(W)
    return Dy @ (R - mu) @ Dx.T, (mu, Dy, Dx)


def fr_normalized(R):
    X, _ = fr_dct(R)
    return FR_E * X / np.linalg.norm(X)


def fr_bands(H, W, nlow, mid_from=0.02, mid_to=0.32):
    """Позиції коефіцієнтів, упорядковані за частотою: НЧ - перші nlow без DC,
    СЧ - 30% усіх коефіцієнтів."""
    u, v = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    order = np.argsort(np.hypot(u / H, v / W).ravel(), kind="stable")[1:]
    n = H * W
    return order[:nlow], order[int(mid_from * n): int(mid_to * n)]


def fr_index(t, alpha):
    """Інтервал i=1,2,... (alpha_i = q^(i-1), q=(1+a)/(1-a)) та ind(t) = (-1)^i.
    |t|<1 відносимо до першого інтервалу."""
    q = (1 + alpha) / (1 - alpha)
    i = np.floor(np.log(np.maximum(np.abs(t), 1.0)) / np.log(q)).astype(int) + 1
    return i, np.where(i % 2 == 0, 1, -1)


def fr_targets(t, Wj, alpha, m=0.3):
    """Нові (нормовані) значення НЧ-коефіцієнтів t так, щоб ind(t_new) = Wj.
    Значення зсувається мінімально - у внутрішню частину потрібного інтервалу
    (відступ m від країв, щоб витримати невеликий зсув масштабу). Напрямок зсуву
    (вгору/вниз) обирається жадібно так, щоб сумарна енергія змінювалася якомога менше."""
    q = (1 + alpha) / (1 - alpha)
    a = np.maximum(np.abs(t), 1.0)
    i, ind = fr_index(t, alpha)
    new = np.empty_like(a)
    diff = 0.0
    for p in np.argsort(-a):
        cands = [int(i[p])] if ind[p] == Wj[p] else [k for k in (int(i[p]) - 1, int(i[p]) + 1) if k >= 1]
        best = None
        for k in cands:
            lo, hi = q ** (k - 1), q ** k
            val = float(np.clip(a[p], lo + m * (hi - lo), hi - m * (hi - lo)))
            d = abs(diff + val ** 2 - a[p] ** 2)
            if best is None or d < best[0]:
                best = (d, val)
        new[p] = best[1]
        diff += best[1] ** 2 - a[p] ** 2
    return np.sign(np.where(t == 0, 1.0, t)) * new


def fr_spread(symbols, nmid, maxw, key):
    """(3.12): сигнал з розширеним спектром (приблизно N(0,1))."""
    rng = np.random.default_rng(key)
    seqs = [rng.random(nmid + maxw) for _ in symbols]
    tot = sum(seqs[j][w - 1: w - 1 + nmid] for j, w in enumerate(symbols))
    L = len(symbols)
    return (tot - L / 2) / np.sqrt(L / 12)


def fr_embed(R, bits, symbols, alpha, gamma, maxw, key):
    """Вбудовування. Приймач нормує зображення на власну енергію (3.7), тому передавач
    зберігає повну енергію ДКП-вектора: НЧ-значення масштабуються спільним множником lam
    так, щоб енергія не змінилася (lam близьке до 1, у межах запасу alpha)."""
    X, (mu, Dy, Dx) = fr_dct(R)
    H, W = R.shape
    low, mid = fr_bands(H, W, len(bits))
    flat = X.ravel().copy()
    E = float(np.sum(flat ** 2))
    kappa = FR_E / np.sqrt(E)                       # фізичні -> нормовані одиниці
    Wj = np.where(np.asarray(bits) == 1, 1, -1)
    if len(symbols):
        flat[mid] += gamma * fr_spread(symbols, len(mid), maxw, key) / kappa
    e_rest = float(np.sum(flat ** 2) - np.sum(X.ravel()[low] ** 2))
    v = fr_targets(X.ravel()[low] * kappa, Wj, alpha)
    target = E - e_rest                             # енергія, що лишається для НЧ
    if target <= 0:
        raise ValueError("Недостатньо енергії для НЧ-коефіцієнтів: зменшіть --fr-bits або gamma")
    lam = np.sqrt(target / (np.sum(v ** 2) / kappa ** 2))
    v = lam * v
    ok = np.mean(np.where(fr_index(v, alpha)[1] == Wj, True, False))
    if abs(lam - 1) > 0.8 * alpha or ok < 1:
        print(f"  (alpha={alpha}: масштаб lam={lam:.3f}, коректних НЧ-індексів {ok:.2f})")
    flat[low] = v / kappa
    P = mu + Dy.T @ flat.reshape(H, W) @ Dx
    return to_uint8(P)


def fr_decode(R, nbits, nsym, alpha, maxw, key):
    Cf = fr_normalized(R)
    H, W = R.shape
    low, mid = fr_bands(H, W, nbits)
    flat = Cf.ravel()
    _, ind = fr_index(flat[low], alpha)
    bits = (ind == 1).astype(np.uint8)
    syms = []
    if nsym:
        rng = np.random.default_rng(key)
        x = flat[mid]
        for _ in range(nsym):
            seq = rng.random(len(mid) + maxw)
            corr = [np.dot(x, seq[k: k + len(mid)] - 0.5) for k in range(maxw)]
            syms.append(int(np.argmax(corr)) + 1)
    return bits, syms


def fr_detect(R, W_pm1, alpha, beta=0.75, delta=0.25, steps=9):
    """K (3.10) при d=1 та K' (3.11) = max по масштабу d."""
    Cf = fr_normalized(R)
    low, _ = fr_bands(*R.shape, len(W_pm1))
    lam = Cf.ravel()[low]
    best, K0 = -2.0, None
    for d in np.linspace(1 - delta, 1 + delta, steps):
        x = lam * d
        _, ind = fr_index(x, alpha)
        w = np.abs(x) ** beta
        K = float(np.sum(w * ind * W_pm1) / np.sum(w))
        if abs(d - 1) < 1e-9:
            K0 = K
        best = max(best, K)
    return K0, best


def task5(rgb, R, bits, quality, out, alpha, gamma, nbits, maxw=16, key=12345):
    print(f"\n=== Завдання 5 (додаткове): метод Дж. Фрідріх, JPEG якість {quality} ===")
    bits_lf = bits[:nbits]
    Wpm = np.where(bits_lf == 1, 1, -1)
    raw = bits_to_bytes(bits[:32])                      # перші 4 байти -> 8 символів 1..16
    symbols = [(b >> s & 15) + 1 for b in raw for s in (0, 4)]
    print(f"НЧ-частина: {len(bits_lf)} біт; СЧ-частина (розширення спектра): "
          f"{len(symbols)} символів, maxW'={maxw}")

    def one(a, g):
        R2 = fr_embed(R, bits_lf, symbols, a, g, maxw, key)
        att = jpeg_attack_rgb(with_red(rgb, R2), quality)
        b0, s0 = fr_decode(R2, len(bits_lf), len(symbols), a, maxw, key)
        b1, s1 = fr_decode(att, len(bits_lf), len(symbols), a, maxw, key)
        return dict(po0=err_prob(bits_lf, b0), po1=err_prob(bits_lf, b1),
                    ss0=float(np.mean(np.array(s0) == symbols)),
                    ss1=float(np.mean(np.array(s1) == symbols)),
                    raz=raz(R, R2), K=fr_detect(att, Wpm, a), R2=R2,
                    Kc=fr_detect(R, Wpm, a))

    r = one(alpha, gamma)
    Image.fromarray(with_red(rgb, r["R2"])).save(os.path.join(out, "Stego_fridrich.bmp"))
    print(f"alpha={alpha}, gamma={gamma}: RAZ = {r['raz']:.3f}")
    print(f"НЧ-біти: Po без атаки = {r['po0']:.3f}, після JPEG = {r['po1']:.3f}")
    print(f"СЧ-символи: частка вірних без атаки = {r['ss0']:.2f}, після JPEG = {r['ss1']:.2f}")
    print(f"Детектування після JPEG: K = {r['K'][0]:.3f}, K' = {r['K'][1]:.3f}; "
          f"чистий контейнер: K = {r['Kc'][0]:.3f}, K' = {r['Kc'][1]:.3f}")
    save_compare(R, r["R2"], "compare_fridrich.png", "Фрідріх", out)

    alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    print(f"\nВплив alpha (gamma={gamma}):")
    print(f"{'alpha':>6} | {'Po(НЧ) JPEG':>11} {'RAZ':>7} {'K після JPEG':>13}")
    rows = []
    for a in alphas:
        q = one(a, gamma)
        rows.append(q)
        print(f"{a:6.2f} | {q['po1']:11.3f} {q['raz']:7.3f} {q['K'][0]:13.3f}")
    gammas = [0.0, 0.5, 1, 2, 3, 4, 6]
    print(f"\nВплив gamma (alpha={alpha}), частка вірних СЧ-символів після JPEG:")
    print(f"{'gamma':>6} | {'вірних':>7} {'RAZ':>7}")
    grows = []
    for g in gammas:
        q = one(alpha, g) if g > 0 else None
        grows.append(q)
        if q:
            print(f"{g:6.1f} | {q['ss1']:7.2f} {q['raz']:7.3f}")

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    axs[0].plot(alphas, [q["po1"] for q in rows], "o-")
    axs[0].set(xlabel="alpha", ylabel="Po", title="НЧ-біти: помилка після JPEG")
    axs[1].plot(alphas, [q["raz"] for q in rows], "o-")
    axs[1].axhline(5.12, color="gray", lw=1)
    axs[1].set(xlabel="alpha", ylabel="RAZ", title="Спотворення контейнера")
    gg = [g for g, q in zip(gammas, grows) if q]
    axs[2].plot(gg, [q["ss1"] for q in grows if q], "o-")
    axs[2].set(xlabel="gamma", ylabel="частка вірних символів", title="СЧ-частина після JPEG")
    for a in axs:
        a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "task5_fridrich.png"), dpi=130)
    plt.close()


class Tee:
    """Дублює вивід у консоль і в лог-файл (завжди UTF-8, незалежно від кодування консолі)."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.o = sys.stdout

    def write(self, t):
        self.o.write(t)
        self.f.write(t)

    def flush(self):
        self.o.flush()
        self.f.flush()


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="1.bmp", help="контейнер (BMP24)")
    ap.add_argument("--msg", default="1.txt", help="текстове повідомлення")
    ap.add_argument("--quality", type=int, default=75, help="якість JPEG для атаки")
    ap.add_argument("--pr", type=int, default=5, help="поріг Pr для завдань 2 і 4")
    ap.add_argument("--PL", type=float, default=200, help="поріг відбракування: великі НЧ-коефіцієнти")
    ap.add_argument("--PH", type=int, default=2, help="поріг відбракування: мінімум ненульових ВЧ-коефіцієнтів")
    ap.add_argument("--alpha", type=float, default=0.1, help="параметр alpha (Фрідріх)")
    ap.add_argument("--gamma", type=float, default=1.0, help="потужність розширення спектра (Фрідріх)")
    ap.add_argument("--fr-bits", type=int, default=40, help="кількість НЧ-біт у методі Фрідріх")
    ap.add_argument("--out", default="results", help="папка для результатів")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out, "run_log.txt"))

    rgb = np.array(Image.open(args.image).convert("RGB"))
    R = rgb[:, :, 0].copy()                                   # READ_RED
    allbits = bytes_to_bits(open(args.msg, "rb").read())
    reject = (args.PL, args.PH)
    cap = capacity(R)
    cap_rj = capacity(R, reject)
    print(f"Контейнер {rgb.shape[1]}x{rgb.shape[0]}, блоків 8x8: {cap}, "
          f"придатних після відбракування (PL={args.PL}, PH={args.PH}): {cap_rj}")
    bits = allbits[:cap_rj]
    if len(bits) < len(allbits):
        print(f"УВАГА: повідомлення обрізано до {len(bits)} біт (місткість)")
    print(f"Повідомлення: {len(bits)} біт, текст: {shown(bits)}")

    task1(R, args.out)
    run_method("Завдання 2: метод Коха-Жао", "kz", rgb, R, bits, args.pr, args.quality, args.out)
    run_method("Завдання 4: метод Бенгама-Мемона-Ео-Юнг (без відбракування)", "bmey",
               rgb, R, bits, args.pr, args.quality, args.out)
    run_method("Завдання 4: Бенгам-Мемон-Ео-Юнг з відбракуванням блоків", "bmey",
               rgb, R, bits, args.pr, args.quality, args.out, reject=reject, tag="bmey_reject")
    task3(rgb, R, bits, args.quality, reject, args.out, list(range(5, 55, 5)))
    task5(rgb, R, bits, args.quality, args.out, args.alpha, args.gamma, min(args.fr_bits, len(bits)))
    print(f"\nГотово. Результати збережено у папці '{args.out}'.")


if __name__ == "__main__":
    main()
