"""Ambiyans seslerini sentezler: kafe, tribün, yağmur, sahil, orman.
Çıktı: ../docs/sesler/<ad>.mp3 (44.1 kHz stereo, uçları birbirine bağlı döngü).
Çalıştır: /opt/miniconda3/bin/python3 sentez.py
"""
import os
import random
import subprocess
import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 44100
SURE = 45          # döngü uzunluğu (sn)
BINDIRME = 3       # döngü dikişindeki çapraz geçiş (sn)
N = SR * (SURE + BINDIRME)
BURA = os.path.dirname(os.path.abspath(__file__))
CIKTI = os.path.join(BURA, "..", "docs", "sesler")
rng = np.random.default_rng(7)
random.seed(7)


def gurultu(n, renk="beyaz"):
    x = rng.standard_normal(n)
    if renk == "pembe":
        b, a = [0.049922, -0.095993, 0.050612, -0.004408], [1, -2.494956, 2.017265, -0.522189]
        x = signal.lfilter(b, a, x)
    elif renk == "kahve":
        x = np.cumsum(x)
        x = signal.lfilter([1, -1], [1, -0.995], x)
    return x / (np.max(np.abs(x)) + 1e-9)


def filtre(x, tur, f, derece=4):
    sos = signal.butter(derece, f, btype=tur, fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def yavas_zarf(n, hiz, alt=0.5):
    """Yavaşça dalgalanan 0..1 zarf."""
    k = max(4, int(n / SR * hiz))
    noktalar = rng.uniform(alt, 1.0, k + 2)
    return np.interp(np.arange(n), np.linspace(0, n, k + 2), noktalar)


def yanki(x, sure=1.2, oran=0.25):
    """Basit oda yankısı: üstel sönen gürültü IR ile konvolüsyon."""
    n = int(SR * sure)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir = filtre(ir, "lowpass", 5000, 2)
    ir /= np.sqrt(np.sum(ir ** 2))
    islak = signal.fftconvolve(x, ir)[: len(x)]
    return x * (1 - oran) + islak * oran * 3


def pan(mono, p):
    """p: -1 sol, +1 sağ (eşit güç)."""
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def ekle(dizi, parca, bas):
    son = min(len(dizi), bas + len(parca))
    if son > bas:
        dizi[bas:son] += parca[: son - bas]


def say_ses(metin, ses, hiz):
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = os.path.join(d, "a.aiff"), os.path.join(d, "a.wav")
        subprocess.run(["say", "-v", ses, "-r", str(hiz), "-o", aiff, metin], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-ar", str(SR), "-ac", "1", wav], check=True)
        _, x = wavfile.read(wav)
    return x.astype(np.float64) / 32768


CUMLELER = [
    "Dün akşam ne oldu biliyor musun, hiç tahmin edemezsin.", "Yok artık, ciddi misin sen?",
    "Bence de öyle, ben de aynısını düşündüm.", "Kahveni nasıl içiyorsun, şekerli mi?",
    "Hafta sonu bir yere gidelim mi diyorum.", "O kadar güldük ki anlatamam.",
    "Patron yine toplantıyı uzattı, bitmek bilmedi.", "Şu tatlıdan bir tane daha alalım mı?",
    "Annem aradı, akşama bize gelecekmiş.", "Ay inanmıyorum, gerçekten mi?",
    "Bu şarkıyı çok seviyorum aslında.", "Yarın erken kalkmam lazım, ondan.",
    "So what did she say after that?", "I told him exactly that, word for word.",
    "We should really come here more often.", "No way, that's hilarious.",
]
SESLER = ["Yelda", "Samantha", "Daniel", "Flo (İngilizce (ABD))", "Reed (İngilizce (ABD))",
          "Eddy (İngilizce (BK))", "Rocko (İngilizce (ABD))", "Grandma (İngilizce (BK))"]


def konusma_katmani(ses_sayisi, hiz_araligi, yuksek=1.0):
    st = np.zeros((N, 2))
    for i in range(ses_sayisi):
        ses = SESLER[i % len(SESLER)]
        hat = np.zeros(N)
        t = int(rng.uniform(0, 3) * SR)
        while t < N:
            metin = " ".join(random.sample(CUMLELER, 2))
            if ses != "Yelda" and random.random() < 0.6:
                metin = " ".join(c for c in CUMLELER if c.isascii())
            x = say_ses(metin, ses, random.randint(*hiz_araligi))
            ekle(hat, x * rng.uniform(0.5, 1.0), t)
            t += len(x) + int(rng.uniform(0.4, 2.5) * SR)
        hat = filtre(hat, "lowpass", rng.uniform(1600, 2600))
        st += pan(hat * yuksek * rng.uniform(0.4, 1.0), rng.uniform(-0.9, 0.9))
    return st


def tink(frekans):
    n = int(SR * 0.6)
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * frekans * k * t) * np.exp(-t * (18 + 10 * j))
            for j, k in enumerate([1, 2.76, 5.4]))
    return x * 0.3


def kafe():
    st = konusma_katmani(9, (150, 200))
    st = np.stack([yanki(st[:, 0], 1.0, 0.35), yanki(st[:, 1], 1.0, 0.35)], axis=1)
    # bardak, kaşık
    t = 0
    while t < N:
        ekle(st, pan(tink(rng.uniform(1800, 3200)) * rng.uniform(0.2, 0.6), rng.uniform(-0.8, 0.8)), t)
        t += int(rng.uniform(0.8, 4) * SR)
    # espresso makinesi
    for bas in rng.uniform(4, SURE - 8, 2):
        n = int(SR * 6)
        x = filtre(gurultu(n), "bandpass", [2500, 7000]) * np.hanning(n) * 0.12
        ekle(st, pan(x, 0.6), int(bas * SR))
    st += pan(filtre(gurultu(N, "kahve"), "lowpass", 300) * 0.05, 0)  # oda uğultusu
    return st


def tribun():
    st = np.zeros((N, 2))
    # kalabalık uğultusu: bant geçiren pembe gürültü, yavaş kabarmalar
    for k in range(6):
        x = filtre(gurultu(N, "pembe"), "bandpass", [250 + 60 * k, 1800 + 200 * k])
        st += pan(x * yavas_zarf(N, 0.25, 0.35) * 0.5, rng.uniform(-0.9, 0.9))
    st += konusma_katmani(6, (210, 260), 0.35)
    # tezahürat: hep bir ağızdan, ritmik
    tez = np.zeros(N)
    for bas in np.arange(3, SURE - 6, 13.0):
        x = say_ses("Haydi! Haydi! Haydi! Haydi!", "Yelda", 200)
        for j in range(12):
            gec = int(rng.uniform(0, 0.12) * SR)
            ekle(tez, x * 0.25 * rng.uniform(0.6, 1.1), int(bas * SR) + gec)
    tez = yanki(filtre(tez, "lowpass", 2200), 2.5, 0.6)
    st += pan(tez, -0.2) * 0.6 + pan(tez, 0.3) * 0.6
    # alkış patlamaları
    for bas in rng.uniform(1, SURE - 4, 5):
        n = int(SR * rng.uniform(1.5, 3))
        alkis = np.zeros(n)
        for _ in range(int(n / SR * 200)):
            p = int(rng.uniform(0, n - 600))
            alkis[p:p + 600] += rng.standard_normal(600) * np.exp(-np.linspace(0, 8, 600))
        alkis = filtre(alkis, "highpass", 900) * np.hanning(n) * 0.06
        ekle(st, pan(alkis, rng.uniform(-0.5, 0.5)), int(bas * SR))
    # uzak düdük
    for bas in rng.uniform(5, SURE - 3, 2):
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        x = np.sin(2 * np.pi * 3100 * t + 6 * np.sin(2 * np.pi * 35 * t)) * np.hanning(n) * 0.05
        ekle(st, pan(yanki(x, 1.5, 0.7), 0.4), int(bas * SR))
    return st


def yagmur():
    st = np.stack([filtre(gurultu(N, "pembe"), "bandpass", [400, 9000]) for _ in range(2)], axis=1) * 0.35
    for _ in range(int(SURE * 120)):
        n = int(SR * rng.uniform(0.005, 0.02))
        d = rng.standard_normal(n) * np.exp(-np.linspace(0, 9, n)) * rng.uniform(0.05, 0.3)
        ekle(st, pan(filtre(d, "highpass", rng.uniform(1500, 5000), 2), rng.uniform(-1, 1)), int(rng.uniform(0, N)))
    st += pan(filtre(gurultu(N, "kahve"), "lowpass", 120) * yavas_zarf(N, 0.1, 0.2) * 0.25, 0)
    return st


def sahil():
    st = np.zeros((N, 2))
    t = 0.0
    while t < SURE + BINDIRME:
        boy = rng.uniform(7, 11)
        n = int(SR * boy)
        zarf = np.concatenate([np.linspace(0, 1, int(n * 0.35)) ** 2, np.exp(-np.linspace(0, 4, n - int(n * 0.35)))])
        for kanal, p in ((0, rng.uniform(-0.8, -0.1)), (1, rng.uniform(0.1, 0.8))):
            x = filtre(gurultu(n, "pembe"), "lowpass", rng.uniform(1500, 3500)) * zarf
            ekle(st, pan(x * 0.7, p), int(t * SR))
        t += boy * rng.uniform(0.6, 0.8)
    st += pan(filtre(gurultu(N), "highpass", 4000) * 0.03, 0)
    st += pan(filtre(gurultu(N, "kahve"), "lowpass", 400) * yavas_zarf(N, 0.2) * 0.12, 0)  # rüzgâr
    return st


def kus():
    tur = rng.integers(3)
    n = int(SR * rng.uniform(0.08, 0.25))
    t = np.arange(n) / SR
    f0 = rng.uniform(2500, 5000)
    if tur == 0:
        f = f0 + 1500 * np.sin(2 * np.pi * rng.uniform(10, 30) * t)
    elif tur == 1:
        f = np.linspace(f0, f0 * rng.uniform(0.5, 1.6), n)
    else:
        f = f0 * (1 + 0.3 * np.sin(2 * np.pi * 60 * t))
    faz = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(faz) * np.hanning(n)


def orman():
    st = np.zeros((N, 2))
    for k in range(2):
        x = filtre(gurultu(N, "pembe"), "bandpass", [150, 1500]) * yavas_zarf(N, 0.15, 0.15)
        st[:, k] += x * 0.3
    t = 0
    while t < N:
        p, yk = rng.uniform(-1, 1), rng.uniform(0.05, 0.25)
        for _ in range(rng.integers(1, 6)):  # kısa ötüş dizisi
            c = kus()
            ekle(st, pan(yanki(c, 0.8, 0.4) * yk, p), t)
            t += len(c) + int(rng.uniform(0.03, 0.15) * SR)
        t += int(rng.uniform(0.5, 4) * SR)
    return st


def dongu_yap(st):
    """Son BINDIRME saniyeyi başa çapraz geçişle bindir; uçlar dikişsiz birleşir."""
    b = BINDIRME * SR
    govde = st[: SURE * SR].copy()
    kuyruk = st[SURE * SR:]
    rampa = np.linspace(0, 1, b)[:, None]
    govde[:b] = govde[:b] * np.sqrt(rampa) + kuyruk * np.sqrt(1 - rampa)
    return govde


def yaz(ad, st):
    st = dongu_yap(st)
    st = st / (np.max(np.abs(st)) + 1e-9) * 0.7
    os.makedirs(CIKTI, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "x.wav")
        wavfile.write(wav, SR, (st * 32767).astype(np.int16))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-b:a", "128k",
                        os.path.join(CIKTI, ad + ".mp3")], check=True)
    print(ad, "tamam")


if __name__ == "__main__":
    for ad, f in [("kafe", kafe), ("tribun", tribun), ("yagmur", yagmur), ("sahil", sahil), ("orman", orman)]:
        yaz(ad, f())
