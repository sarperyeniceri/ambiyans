"""Ambiyans seslerini sentezler.
Her ortam iki katmandan oluşur:
  - yatak: belirgin olayı olmayan, sürekli dönen iki uğultu kaydı (farklı uzunlukta; üst üste
    binince desen ancak çok uzun sürede tekrar eder)
  - olaylar: garson geçişi, fincan, devrilen tepsi gibi tek seferlik sesler; sayfa bunları
    rastgele zamanda, rastgele yerden çalar (iki telefonda aynı tohumla, aynı anda).
Çıktı: ../docs/sesler/<ortam>/*.mp3 ve ../docs/sesler/manifest.json
Çalıştır: /opt/miniconda3/bin/python3 sentez.py
"""
import functools
import json
import os
import random
import shutil
import subprocess
import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 44100
BINDIRME = 3                 # yatak döngüsündeki çapraz geçiş (sn)
YATAK_SURELERI = (43, 67)    # iki yatak, ortak katları uzak
BURA = os.path.dirname(os.path.abspath(__file__))
CIKTI = os.path.join(BURA, "..", "docs", "sesler")
rng = np.random.default_rng(11)
random.seed(11)


def n_(sn):
    return int(SR * sn)


def zaman(n):
    return np.arange(n) / SR


# ---------- temel araçlar ----------
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
    return signal.sosfilt(signal.butter(derece, f, btype=tur, fs=SR, output="sos"), x)


def yavas_zarf(n, hiz, alt=0.5):
    k = max(4, int(n / SR * hiz))
    return np.interp(np.arange(n), np.linspace(0, n, k + 2), rng.uniform(alt, 1.0, k + 2))


def yanki(x, sure=1.2, oran=0.25):
    n = n_(sure)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir = filtre(ir, "lowpass", 5000, 2)
    ir /= np.sqrt(np.sum(ir ** 2))
    pad = np.concatenate([x, np.zeros(n)])
    islak = signal.fftconvolve(pad, ir)[: len(pad)]
    return pad * (1 - oran) + islak * oran * 3


def pan(mono, p):
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def ekle(dizi, parca, bas):
    bas = max(0, bas)
    son = min(len(dizi), bas + len(parca))
    if son > bas:
        dizi[bas:son] += parca[: son - bas]


def zarf_ad(n, atak, sonum):
    """Atak + üstel sönüm."""
    a = n_(atak)
    z = np.exp(-np.linspace(0, sonum, n))
    if a:
        z[:a] *= np.linspace(0, 1, a)
    return z


def kismi(frekanslar, sonumler, sure, genlikler=None):
    t = zaman(n_(sure))
    genlikler = genlikler or [1] * len(frekanslar)
    return sum(g * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t * s)
               for f, s, g in zip(frekanslar, sonumler, genlikler))


@functools.lru_cache(maxsize=None)
def say_ses(metin, ses="Yelda", hiz=180):
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = os.path.join(d, "a.aiff"), os.path.join(d, "a.wav")
        subprocess.run(["say", "-v", ses, "-r", str(hiz), "-o", aiff, metin], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-ar", str(SR), "-ac", "1", wav], check=True)
        _, x = wavfile.read(wav)
    return x.astype(np.float64) / 32768


def kalabalik_unlu(sure, formant, kisi=30, f0=(110, 260), egim=0.85, atak=0.25):
    """Kalabalığın hep bir ağızdan 'oooo' / 'uuuu' demesi: testere dişi sesler + formant süzgeci."""
    n = n_(sure)
    t = zaman(n)
    x = np.zeros(n)
    for _ in range(kisi):
        f = rng.uniform(*f0) * np.linspace(1, egim * rng.uniform(0.95, 1.05), n)
        f *= 1 + 0.012 * np.sin(2 * np.pi * rng.uniform(4, 6.5) * t + rng.uniform(0, 6))
        gecik = n_(rng.uniform(0, 0.25))
        ses = signal.sawtooth(2 * np.pi * np.cumsum(f) / SR) * rng.uniform(0.4, 1)
        ses[:gecik] = 0
        x += ses
    x = sum(filtre(x, "bandpass", [fm * 0.85, fm * 1.15], 2) * g for fm, g in formant)
    x += filtre(gurultu(n), "bandpass", [300, 2500]) * 0.15
    return x * zarf_ad(n, atak, 2.5) / (np.max(np.abs(x)) + 1e-9)


def alkis(sure, yogunluk=150):
    n = n_(sure)
    x = np.zeros(n)
    for _ in range(int(sure * yogunluk)):
        p = int(rng.uniform(0, n - 700))
        x[p:p + 700] += rng.standard_normal(700) * np.exp(-np.linspace(0, 9, 700)) * rng.uniform(0.3, 1)
    return filtre(x, "bandpass", [900, 6000]) * np.hanning(n)


def tink(f, sure=0.5, sonum=18):
    return kismi([f, f * 2.76, f * 5.4], [sonum, sonum * 1.6, sonum * 2.4], sure, [1, 0.5, 0.25])


def cam_kirilma():
    n = n_(1.6)
    x = filtre(gurultu(n), "highpass", 2500) * np.exp(-zaman(n) * 25) * 0.6
    for _ in range(45):
        bas = n_(rng.exponential(0.18))
        parca = tink(rng.uniform(2500, 9000), rng.uniform(0.05, 0.25), rng.uniform(25, 60)) * rng.uniform(0.1, 0.5)
        ekle(x, parca, bas)
    return x


def tepsi_carpma():
    """Metal tepsi düşer, sekip döner (vuruş aralıkları kısalır)."""
    f0 = rng.uniform(330, 480)
    oran = [1, 2.32, 4.25, 6.63, 9.1]
    n = n_(3.2)
    x = np.zeros(n)
    t, ara, g = 0.0, 0.32, 1.0
    while ara > 0.008 and t < 2.4:
        vurus = kismi([f0 * k * rng.uniform(0.99, 1.01) for k in oran], [3, 4, 6, 9, 13], 1.2, [1, .7, .5, .35, .2])
        vurus += filtre(gurultu(len(vurus)), "highpass", 1500) * np.exp(-zaman(len(vurus)) * 60) * 0.6
        ekle(x, vurus * g, n_(t))
        t += ara
        ara *= 0.72 if t > 0.5 else 0.6
        g *= 0.82
    return x


def adimlar(adet, aralik, sertlik=1.0):
    n = n_(adet * aralik + 0.5)
    x = np.zeros(n)
    for i in range(adet):
        topuk = filtre(gurultu(n_(0.08)), "lowpass", 900) * np.exp(-np.linspace(0, 9, n_(0.08)))
        tik = filtre(gurultu(n_(0.02)), "highpass", 2500) * np.exp(-np.linspace(0, 8, n_(0.02))) * 0.4 * sertlik
        bas = n_(i * aralik + rng.uniform(-0.03, 0.03))
        ekle(x, topuk * rng.uniform(0.6, 1), bas)
        ekle(x, tik, bas + n_(0.01))
    return x


def kus():
    tur = rng.integers(3)
    n = n_(rng.uniform(0.08, 0.25))
    t = zaman(n)
    f0 = rng.uniform(2500, 5000)
    if tur == 0:
        f = f0 + 1500 * np.sin(2 * np.pi * rng.uniform(10, 30) * t)
    elif tur == 1:
        f = np.linspace(f0, f0 * rng.uniform(0.5, 1.6), n)
    else:
        f = f0 * (1 + 0.3 * np.sin(2 * np.pi * 60 * t))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(n)


def ruzgar_savrulma(sure):
    n = n_(sure)
    x = filtre(gurultu(n, "pembe"), "bandpass", [rng.uniform(150, 300), rng.uniform(1200, 2500)])
    return x * np.hanning(n) ** 1.5


# ---------- yatak (sürekli katman) ----------
def dongu_yap(st, sure):
    b = n_(BINDIRME)
    govde = st[: n_(sure)].copy()
    kuyruk = st[n_(sure): n_(sure) + b]
    r = np.linspace(0, 1, b)[:, None]
    govde[:b] = govde[:b] * np.sqrt(r) + kuyruk * np.sqrt(1 - r)
    return govde


CUMLELER = [
    "Dün akşam ne oldu biliyor musun", "Yok artık ciddi misin", "Bence de öyle", "Kahveni nasıl içiyorsun",
    "Hafta sonu bir yere gidelim", "O kadar güldük ki", "Patron yine toplantıyı uzattı", "Şu tatlıdan alalım mı",
    "Annem aradı akşama gelecekmiş", "Gerçekten mi", "Yarın erken kalkmam lazım", "Ne diyorsun sen",
    "Aynen öyle oldu", "Sonra ne dedi", "Hiç sorma", "Valla bilmiyorum", "Tamam tamam anladım", "Çok iyi ya",
]
YABANCI = ["So what did she say", "No way", "We should come here more often", "That's hilarious",
           "I told him exactly that", "Are you serious"]
SESLER = ["Yelda", "Samantha", "Daniel", "Flo (İngilizce (ABD))", "Reed (İngilizce (ABD))",
          "Eddy (İngilizce (BK))", "Rocko (İngilizce (ABD))", "Grandma (İngilizce (BK))"]


def ugultu(n, kisi, hiz, lp):
    """Anlaşılmayan kalabalık konuşması: kısa cümleler, çok kişi, kısık ve boğuk."""
    st = np.zeros((n, 2))
    for i in range(kisi):
        ses = SESLER[i % len(SESLER)]
        havuz = CUMLELER if ses == "Yelda" else YABANCI
        hat = np.zeros(n)
        t = n_(rng.uniform(0, 4))
        while t < n:
            x = say_ses(random.choice(havuz), ses, random.choice(hiz))
            ekle(hat, x * rng.uniform(0.4, 1.0), t)
            t += len(x) + n_(rng.uniform(0.6, 3.0))
        st += pan(filtre(hat, "lowpass", lp * rng.uniform(0.8, 1.2)) * rng.uniform(0.3, 1.0), rng.uniform(-0.95, 0.95))
    return st


def stereo_yanki(st, sure, oran):
    a, b = yanki(st[:, 0], sure, oran), yanki(st[:, 1], sure, oran)
    return np.stack([a, b], axis=1)[: len(st)]


def yatak_kafe(n):
    st = stereo_yanki(ugultu(n, 16, (160, 185, 210), 1500), 1.0, 0.45)
    st += pan(filtre(gurultu(n, "kahve"), "lowpass", 250), 0) * 0.04
    return st


def yatak_tribun(n):
    st = np.zeros((n, 2))
    for k in range(6):
        x = filtre(gurultu(n, "pembe"), "bandpass", [250 + 60 * k, 1800 + 200 * k])
        st += pan(x * yavas_zarf(n, 0.2, 0.55) * 0.5, rng.uniform(-0.9, 0.9))
    st += stereo_yanki(ugultu(n, 10, (220, 250), 1100), 2.0, 0.6) * 0.5
    return st


def yatak_yagmur(n):
    st = np.stack([filtre(gurultu(n, "pembe"), "bandpass", [400, 9000]) for _ in range(2)], axis=1) * 0.35
    for _ in range(int(n / SR * 140)):
        m = n_(rng.uniform(0.005, 0.02))
        d = rng.standard_normal(m) * np.exp(-np.linspace(0, 9, m)) * rng.uniform(0.05, 0.25)
        ekle(st, pan(filtre(d, "highpass", rng.uniform(1500, 5000), 2), rng.uniform(-1, 1)), int(rng.uniform(0, n)))
    st += pan(filtre(gurultu(n, "kahve"), "lowpass", 120) * yavas_zarf(n, 0.1, 0.4) * 0.2, 0)
    return st


def yatak_sahil(n):
    st = np.stack([filtre(gurultu(n, "pembe"), "lowpass", 700) * yavas_zarf(n, 0.15, 0.6) for _ in range(2)], axis=1) * 0.5
    st += np.stack([filtre(gurultu(n), "highpass", 4000) for _ in range(2)], axis=1) * 0.02
    st += pan(filtre(gurultu(n, "kahve"), "lowpass", 400) * yavas_zarf(n, 0.2) * 0.15, 0)
    return st


def yatak_orman(n):
    st = np.stack([filtre(gurultu(n, "pembe"), "bandpass", [900, 6000]) * yavas_zarf(n, 0.3, 0.3) for _ in range(2)], axis=1) * 0.25
    st += np.stack([filtre(gurultu(n, "pembe"), "bandpass", [120, 900]) * yavas_zarf(n, 0.12, 0.2) for _ in range(2)], axis=1) * 0.3
    return st


# ---------- olaylar ----------
def garson(cumle):
    adim = adimlar(8, rng.uniform(0.42, 0.55), 1.3)
    for i in range(8):  # tepsideki bardaklar şıngırdar
        ekle(adim, tink(rng.uniform(2200, 3800), 0.3, 30) * 0.08, n_(i * 0.5 + 0.05))
    if cumle:
        ses = filtre(say_ses(cumle, "Yelda", 185), "lowpass", 3500) * 0.9
        ekle(adim, ses, n_(rng.uniform(1.2, 1.8)))
    return adim


def devirme():
    n = n_(7)
    x = np.zeros(n)
    ekle(x, tepsi_carpma(), 0)
    ekle(x, cam_kirilma() * 0.9, n_(0.03))
    ekle(x, kalabalik_unlu(2.2, [(450, 1), (800, 0.6)], 25, (120, 240), 0.8) * 0.25, n_(0.55))
    ekle(x, alkis(2.5, 25) * 1.4, n_(2.6))  # birkaç kişi takılır: alkış
    return x


def sandalye():
    n = n_(rng.uniform(0.35, 0.9))
    hiz = rng.uniform(50, 130)
    darbe = (rng.uniform(0, 1, n) < hiz / SR).astype(float) * rng.uniform(0.5, 1, n)
    f = rng.uniform(500, 1300)
    x = filtre(darbe, "bandpass", [f * 0.9, f * 1.1], 2) * 6 + filtre(gurultu(n), "bandpass", [300, 3000]) * 0.15
    return x * np.hanning(n) ** 0.5


def kasik_karistirma():
    x = np.zeros(n_(2.5))
    for i in range(rng.integers(5, 9)):
        ekle(x, tink(rng.uniform(3200, 4200), 0.2, 35) * rng.uniform(0.3, 0.6), n_(i * 0.27 + rng.uniform(0, 0.03)))
    return x


def fincan_tabak():
    x = tink(rng.uniform(1100, 1700), 0.6, 14)
    x[: n_(0.04)] += filtre(gurultu(n_(0.04)), "lowpass", 600) * np.exp(-np.linspace(0, 6, n_(0.04)))
    return x


def espresso():
    n = n_(9)
    x = np.zeros(n)
    t = zaman(n_(3))
    motor = signal.sawtooth(2 * np.pi * 155 * t) * 0.3 + filtre(gurultu(len(t)), "bandpass", [700, 3000])
    motor *= (1 + 0.3 * np.sin(2 * np.pi * 23 * t)) * np.hanning(len(t)) ** 0.2
    ekle(x, motor * 0.6, 0)
    m = n_(5)
    buhar = filtre(gurultu(m), "bandpass", [2500, 9000]) * (0.7 + 0.3 * filtre(rng.standard_normal(m), "lowpass", 15) * 8)
    ekle(x, buhar * np.hanning(m) * 0.8, n_(3.8))
    return x


def kapi_zili():
    x = np.zeros(n_(2.5))
    f0 = rng.uniform(1700, 2300)
    for i in range(rng.integers(3, 6)):
        ekle(x, kismi([f0, f0 * 1.5, f0 * 2.1, f0 * 2.9], [4, 5, 7, 9], 1.8, [1, .5, .4, .2]) * rng.uniform(0.4, 1), n_(rng.uniform(0, 0.5)))
    ekle(x, filtre(gurultu(n_(0.15)), "lowpass", 300) * np.exp(-np.linspace(0, 6, n_(0.15))) * 0.8, n_(1.2))
    return x


def yakin_masa(metin, ses):
    return filtre(say_ses(metin, ses, rng.integers(165, 200)), "lowpass", 2800) * 0.8


def tezahurat(metin, tekrar=2):
    tek = say_ses(metin, "Yelda", 205)
    x = np.zeros(len(tek) * tekrar + n_(1))
    for r in range(tekrar):
        for _ in range(14):
            ekle(x, tek * rng.uniform(0.5, 1.1), r * len(tek) + n_(rng.uniform(0, 0.12)))
    return filtre(x, "lowpass", 2200)


def davul(vurus):
    x = np.zeros(n_(len(vurus) * 0.32 + 1))
    for i, guclu in enumerate(vurus):
        m = n_(0.4)
        t = zaman(m)
        f = 75 * (1 + 0.8 * np.exp(-t * 30))
        tok = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
        tok += filtre(gurultu(m), "lowpass", 1200) * np.exp(-t * 50) * 0.4
        ekle(x, tok * (1 if guclu else 0.55), n_(i * 0.32))
    return x


def hakem_duduk():
    x = np.zeros(n_(1.5))
    for i in range(rng.integers(1, 4)):
        m = n_(rng.uniform(0.15, 0.6))
        t = zaman(m)
        s = np.sin(2 * np.pi * 3100 * t + 6 * np.sin(2 * np.pi * 35 * t)) * np.hanning(m) ** 0.3
        ekle(x, s * 0.5, n_(i * 0.35))
    return x


def islik_yuhalama():
    n = n_(4)
    x = kalabalik_unlu(4, [(300, 1), (870, 0.4)], 30, (100, 200), 0.95, 0.6) * 0.7
    for _ in range(25):
        m = n_(rng.uniform(0.5, 2.5))
        t = zaman(m)
        f = rng.uniform(2200, 3800) * (1 + 0.02 * np.sin(2 * np.pi * rng.uniform(4, 8) * t))
        ekle(x, np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(m) * 0.04, n_(rng.uniform(0, 2)))
    return x[:n]


def gol():
    n = n_(10)
    x = np.zeros(n)
    kuk = filtre(gurultu(n, "pembe"), "bandpass", [250, 3000])
    z = np.concatenate([np.linspace(0.1, 1, n_(0.8)), np.exp(-np.linspace(0, 2, n - n_(0.8)))])
    ekle(x, kuk * z, 0)
    ekle(x, kalabalik_unlu(6, [(500, 1), (900, 0.7)], 40, (150, 320), 0.9, 0.5) * 0.8, n_(0.3))
    ekle(x, davul([1, 0] * 10) * 0.5, n_(1.5))
    return x


def korna():
    t = zaman(n_(rng.uniform(0.8, 1.6)))
    f = rng.uniform(330, 420)
    s = signal.sawtooth(2 * np.pi * f * t) + signal.sawtooth(2 * np.pi * f * 1.006 * t)
    return filtre(s, "lowpass", 2500) * np.hanning(len(t)) ** 0.2 * 0.4


def gok_gurultusu(yakin):
    n = n_(rng.uniform(5, 9))
    x = filtre(gurultu(n, "kahve"), "lowpass", 250 if yakin else 150)
    yuvarlan = np.abs(filtre(rng.standard_normal(n), "lowpass", 3)) * 30 + 0.2
    x *= yuvarlan * zarf_ad(n, 0.05 if yakin else 0.6, 3.5)
    if yakin:
        ekle(x, filtre(gurultu(n_(0.25)), "highpass", 800) * np.exp(-np.linspace(0, 5, n_(0.25))) * 0.8, 0)
    return x


def araba_gecisi():
    n = n_(rng.uniform(4, 6))
    t = np.linspace(-1, 1, n)
    x = filtre(gurultu(n), "bandpass", [400, 5000]) * np.exp(-(t / 0.45) ** 2)
    return x + filtre(gurultu(n, "kahve"), "lowpass", 150) * np.exp(-(t / 0.5) ** 2) * 0.5


def oluk_damla():
    x = np.zeros(n_(9))
    t = 0.0
    while t < 8:
        ekle(x, tink(rng.uniform(900, 1800), 0.15, 40) * rng.uniform(0.3, 0.8), n_(t))
        t += rng.uniform(0.25, 0.9)
    return x * 0.6


def dalga():
    n = n_(rng.uniform(6, 10))
    z = np.concatenate([np.linspace(0, 1, int(n * 0.35)) ** 2, np.exp(-np.linspace(0, 4, n - int(n * 0.35)))])
    return filtre(gurultu(n, "pembe"), "lowpass", rng.uniform(1500, 3500)) * z


def marti():
    x = np.zeros(n_(2.5))
    for i in range(rng.integers(2, 5)):
        m = n_(rng.uniform(0.25, 0.4))
        t = zaman(m)
        f = np.linspace(rng.uniform(1300, 1600), rng.uniform(700, 900), m)
        faz = 2 * np.pi * np.cumsum(f) / SR
        s = sum(np.sin(k * faz) / k ** 0.7 for k in range(1, 7))
        s = filtre(s, "bandpass", [900, 4500]) * np.hanning(m) ** 0.6
        ekle(x, s * rng.uniform(0.6, 1), n_(i * rng.uniform(0.35, 0.5)))
    return x


def vapur():
    t = zaman(n_(3))
    f = 98 * (1 + 0.003 * np.sin(2 * np.pi * 5 * t))
    s = filtre(signal.sawtooth(2 * np.pi * np.cumsum(f) / SR), "lowpass", 1000)
    return s * zarf_ad(len(t), 0.25, 0.5) * np.hanning(len(t)) ** 0.2


def kus_dizisi():
    x = np.zeros(n_(3))
    t = 0
    for _ in range(rng.integers(2, 7)):
        c = kus()
        ekle(x, c, t)
        t += len(c) + n_(rng.uniform(0.03, 0.15))
    return x


def agackakan():
    x = np.zeros(n_(2))
    hiz = rng.uniform(14, 20)
    for i in range(rng.integers(10, 20)):
        m = n_(0.03)
        tok = filtre(gurultu(m), "bandpass", [700, 1600]) * np.exp(-np.linspace(0, 8, m))
        ekle(x, tok * (1 - i * 0.03), n_(i / hiz))
    return x


def guguk():
    x = np.zeros(n_(3.5))
    for i in range(rng.integers(2, 5)):
        for j, f in enumerate([760, 610]):
            m = n_(0.25)
            s = np.sin(2 * np.pi * f * zaman(m)) * np.hanning(m) ** 0.7
            ekle(x, s, n_(i * 0.85 + j * 0.3))
    return x


def dal_kirilmasi():
    x = filtre(gurultu(n_(0.12)), "bandpass", [800, 5000]) * np.exp(-np.linspace(0, 6, n_(0.12)))
    hisir = filtre(gurultu(n_(1.2)), "bandpass", [1500, 7000]) * np.hanning(n_(1.2)) * 0.15
    y = np.zeros(n_(1.5))
    ekle(y, x, 0)
    ekle(y, hisir, n_(0.1))
    return y


# ---------- ek yataklar ----------
def oda_tonu(n, lp=300, g=0.05):
    return np.stack([filtre(gurultu(n, "kahve"), "lowpass", lp) for _ in range(2)], axis=1) * g


def piyano_nota(f, sure=3.0, guc=0.8):
    m = n_(sure)
    t = zaman(m)
    x = sum((guc / k ** 1.3) * np.sin(2 * np.pi * k * f * np.sqrt(1 + 0.0004 * k * k) * t) * np.exp(-t * (0.9 + 0.5 * k) * (f / 260) ** 0.5)
            for k in range(1, 9))
    x[: n_(0.004)] *= np.linspace(0, 1, n_(0.004))
    x[: n_(0.02)] += filtre(gurultu(n_(0.02)), "bandpass", [1000, 4000]) * 0.02 * guc
    return x


AKORLAR = [  # ii-V-I-vi, kök sesler ve yarım ton aralıkları (yumuşak caz voicing)
    [(50, [0, 15, 19, 22, 26])], [(55, [0, 16, 22, 26, 29])], [(48, [0, 16, 19, 23, 26])], [(57, [0, 15, 19, 22, 26])],
]


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


def piyano_yatak(n, transpoz=0, olcu=2.0):
    x = np.zeros(n + n_(4))
    olcu_say = int(n / SR / olcu) + 2
    for o in range(olcu_say):
        kok, araliklar = AKORLAR[o % 4][0]
        kok += transpoz
        bas = n_(o * olcu)
        ekle(x, piyano_nota(midi(kok - 12), 3, 0.6), bas)
        for j, a in enumerate(araliklar[1:]):
            ekle(x, piyano_nota(midi(kok + a), 2.5, 0.35), bas + n_(rng.uniform(0, 0.04) + (j * 0.06 if o % 2 else 0)))
        for b in range(4):  # gezinen ezgi
            if rng.uniform() < 0.55:
                nota = kok + 24 + random.choice(araliklar[1:] + [araliklar[1] + 2, araliklar[2] + 2])
                ekle(x, piyano_nota(midi(nota), 1.5, rng.uniform(0.3, 0.5)), bas + n_(b * olcu / 4 + rng.choice([0, olcu / 8])))
    x = filtre(yanki(x, 1.8, 0.35), "lowpass", 5000)
    return pan(x[:n], -0.3) + pan(x[:n], 0.3) * 0.8


def ates_yatak(n):
    st = np.stack([filtre(gurultu(n, "kahve"), "lowpass", 200) * yavas_zarf(n, 0.5, 0.6) for _ in range(2)], axis=1) * 0.5
    st += np.stack([filtre(gurultu(n), "bandpass", [600, 3000]) * yavas_zarf(n, 1, 0.3) for _ in range(2)], axis=1) * 0.04
    for _ in range(int(n / SR * 25)):  # çıtırtı
        m = n_(rng.uniform(0.002, 0.012))
        d = rng.standard_normal(m) * np.exp(-np.linspace(0, 6, m)) * rng.uniform(0.1, 0.9)
        ekle(st, pan(filtre(d, "highpass", rng.uniform(1000, 3500), 2), rng.uniform(-0.4, 0.4)), int(rng.uniform(0, n)))
    return st


def cekirge_yatak(n, sayi=6):
    st = np.zeros((n, 2))
    for _ in range(sayi):
        f = rng.uniform(3800, 5200)
        nabiz = rng.uniform(25, 45)
        tur = rng.uniform(0.6, 1.4)
        t = zaman(n)
        tasiyici = np.sin(2 * np.pi * f * t)
        darbe = (np.sin(2 * np.pi * nabiz * t) > 0.3).astype(float)
        kapi = (np.sin(2 * np.pi * t / tur + rng.uniform(0, 6)) > rng.uniform(-0.2, 0.4)).astype(float)
        kapi = filtre(kapi, "lowpass", 30, 2)
        st += pan(tasiyici * darbe * kapi * rng.uniform(0.03, 0.12), rng.uniform(-0.9, 0.9))
    return st + oda_tonu(n, 200, 0.03)


def motor_yatak(n, f0, lp, ugultu_g=0.5):
    t = zaman(n)
    hum = sum(np.sin(2 * np.pi * f0 * k * t * (1 + 0.002 * np.sin(2 * np.pi * 0.1 * t))) / k for k in range(1, 6))
    st = pan(hum * 0.15, 0)
    st += np.stack([filtre(gurultu(n, "pembe"), "lowpass", lp) for _ in range(2)], axis=1) * ugultu_g
    return st


def ray_yatak(n):
    st = motor_yatak(n, 48, 600, 0.6)
    for i in range(int(n / SR / 2) + 1):  # her 2 sn'de bir takırtı çifti
        for d in (0, 0.18):
            m = n_(0.1)
            tak = filtre(gurultu(m), "bandpass", [150, 1200]) * np.exp(-np.linspace(0, 7, m)) * 0.9
            ekle(st, pan(tak, -0.2 if d else 0.2), n_(i * 2 + d))
    return st


def saat_tik(n):
    st = np.zeros((n, 2))
    for i in range(int(n / SR)):
        m = n_(0.03)
        tik = filtre(gurultu(m), "bandpass", [2000, 5000]) * np.exp(-np.linspace(0, 9, m)) * (0.12 if i % 2 else 0.09)
        ekle(st, pan(tik, 0.6), n_(i))
    return st


def su_sirildama(n):
    st = np.zeros((n, 2))
    for _ in range(int(n / SR * 40)):
        m = n_(rng.uniform(0.03, 0.1))
        f = rng.uniform(500, 1600)
        t = zaman(m)
        b = np.sin(2 * np.pi * np.cumsum(f * (1 + 0.6 * t / t[-1])) / SR) * np.hanning(m) * rng.uniform(0.05, 0.2)
        ekle(st, pan(b, rng.uniform(-0.5, 0.5)), int(rng.uniform(0, n)))
    return st + np.stack([filtre(gurultu(n, "pembe"), "bandpass", [300, 4000]) for _ in range(2)], axis=1) * 0.08


def su_sapirti(n):
    st = np.zeros((n, 2))
    t = 0
    while t < n:
        m = n_(rng.uniform(0.6, 1.4))
        x = filtre(gurultu(m, "pembe"), "lowpass", rng.uniform(500, 1200)) * np.hanning(m) * rng.uniform(0.3, 0.8)
        ekle(st, pan(x, rng.uniform(-0.8, 0.8)), t)
        t += n_(rng.uniform(0.4, 1.2))
    return st


def kulup_yatak(n):
    x = np.zeros(n + n_(1))
    for i in range(int(n / SR * 2) + 1):  # 120 bpm
        m = n_(0.3)
        t = zaman(m)
        kick = np.sin(2 * np.pi * np.cumsum(50 * (1 + 2 * np.exp(-t * 40))) / SR) * np.exp(-t * 8)
        ekle(x, kick, n_(i * 0.5))
        ekle(x, filtre(gurultu(n_(0.05)), "highpass", 7000) * np.exp(-np.linspace(0, 6, n_(0.05))) * 0.12, n_(i * 0.5 + 0.25))
    bas = [43, 43, 46, 41]
    for o in range(int(n / SR / 2) + 1):
        f = midi(bas[o % 4])
        for k in range(8):
            m = n_(0.2)
            s = signal.sawtooth(2 * np.pi * f * zaman(m)) * np.exp(-zaman(m) * 10) * 0.25
            ekle(x, s, n_(o * 2 + k * 0.25 + 0.125))
    x = filtre(x[:n], "lowpass", 700)
    st = pan(x, 0) * 0.9
    st += ugultu(n, 8, (190, 220), 1300) * 0.35
    return st


def uzay_yatak(n):
    t = zaman(n)
    hum = sum(np.sin(2 * np.pi * 60 * k * t) / k ** 1.5 for k in range(1, 8)) * 0.08
    st = pan(hum, 0) + np.stack([filtre(gurultu(n, "pembe"), "bandpass", [200, 2500]) for _ in range(2)], axis=1) * 0.25
    return st


# ---------- ek olaylar ----------
def kadeh():
    f = rng.uniform(2400, 3400)
    x = kismi([f, f * 1.003, f * 2.4], [3, 3, 6], 2.5, [1, 0.8, 0.3])
    y = np.zeros(n_(2.7))
    ekle(y, x, 0)
    ekle(y, kismi([f * 1.07, f * 2.5], [3, 6], 2.5, [0.7, 0.2]), n_(0.01))
    return y


def sampanya():
    y = np.zeros(n_(3))
    m = n_(0.08)
    ekle(y, np.sin(2 * np.pi * np.cumsum(np.linspace(400, 120, m)) / SR) * np.exp(-np.linspace(0, 6, m)), 0)
    ekle(y, filtre(gurultu(m), "bandpass", [500, 3000]) * np.exp(-np.linspace(0, 8, m)) * 0.8, 0)
    k = n_(2.5)
    kopuk = np.zeros(k)
    for p in rng.integers(0, k, 900):
        kopuk[p] += rng.uniform(-1, 1)
    ekle(y, filtre(kopuk, "highpass", 4000) * np.exp(-zaman(k) * 1.2) * 0.4, n_(0.2))
    return y


def buz_bardak():
    y = np.zeros(n_(1.5))
    for i in range(rng.integers(3, 7)):
        ekle(y, tink(rng.uniform(1400, 2600), 0.3, 25) * rng.uniform(0.3, 0.8), n_(rng.uniform(0, 0.8)))
    return y


def sarap_dokme():
    y = np.zeros(n_(3))
    t = 0.0
    while t < 2.2:
        m = n_(0.09)
        f = 300 + t * 180
        glu = filtre(gurultu(m), "bandpass", [f, f * 2.5], 2) * np.hanning(m)
        ekle(y, glu * rng.uniform(0.5, 1), n_(t))
        t += rng.uniform(0.08, 0.14)
    return y


def catal_bicak():
    y = np.zeros(n_(1.5))
    for i in range(rng.integers(2, 5)):
        ekle(y, tink(rng.uniform(2800, 4500), 0.25, 30) * rng.uniform(0.2, 0.6), n_(rng.uniform(0, 1)))
    m = n_(0.25)
    ekle(y, filtre(gurultu(m), "bandpass", [2000, 6000]) * np.hanning(m) * 0.15, n_(rng.uniform(0, 1)))
    return y


def asansor():
    y = np.zeros(n_(2.5))
    ekle(y, kismi([1318], [2.5], 1.5) * 0.6, 0)
    ekle(y, filtre(gurultu(n_(1.2)), "bandpass", [200, 1500]) * np.hanning(n_(1.2)) * 0.15, n_(0.6))
    return y


def resepsiyon_zili():
    return kismi([2650, 2650 * 2.7, 2650 * 5.1], [1.5, 3, 5], 2.5, [1, 0.3, 0.1])


def bavul():
    n = n_(5)
    x = filtre(gurultu(n), "bandpass", [100, 900]) * (1 + 0.5 * np.sin(2 * np.pi * 9 * zaman(n))) * 0.4
    for i in range(10):
        ekle(x, filtre(gurultu(n_(0.02)), "bandpass", [400, 2000]) * 0.6, n_(i * 0.5 + 0.2))
    return x * np.hanning(n) + adimlar(9, 0.55, 1.2)[:n] * 0.6


def anons(metin, ding=True):
    y = np.zeros(n_(1.4))
    if ding:
        ekle(y, kismi([659], [2], 1.2) * 0.5, 0)
        ekle(y, kismi([523], [2], 1.2) * 0.5, n_(0.5))
    s = filtre(say_ses(metin, "Yelda", 175), "bandpass", [350, 3400]) * 0.9
    return np.concatenate([y, s])


def slot():
    y = np.zeros(n_(3))
    notlar = [72, 76, 79, 84, 79, 84, 88]
    for i, m_ in enumerate(notlar[: rng.integers(4, 8)]):
        m = n_(0.12)
        s = signal.square(2 * np.pi * midi(m_) * zaman(m)) * np.hanning(m) * 0.2
        ekle(y, filtre(s, "lowpass", 4000), n_(i * 0.1))
    if rng.uniform() < 0.4:
        for i in range(25):
            ekle(y, tink(rng.uniform(3000, 5000), 0.2, 30) * 0.3, n_(0.8 + i * 0.06 + rng.uniform(0, 0.03)))
    return y


def rulet():
    n = n_(6)
    t = zaman(n)
    hiz = 6 * np.exp(-t * 0.35)
    x = filtre(gurultu(n), "bandpass", [800, 3000]) * (0.6 + 0.4 * np.sin(2 * np.pi * np.cumsum(hiz) / SR)) * 0.3
    x *= np.exp(-t * 0.25)
    for i in range(6):
        ekle(x, filtre(gurultu(n_(0.02)), "highpass", 2000) * (0.7 - i * 0.1), n_(4.6 + i * rng.uniform(0.08, 0.2)))
    return x


def fis():
    y = np.zeros(n_(1))
    for i in range(rng.integers(3, 9)):
        ekle(y, tink(rng.uniform(1100, 1600), 0.12, 45) * 0.5, n_(i * rng.uniform(0.03, 0.07)))
    return y


def gulusme(kisi=4):
    n = n_(rng.uniform(1.2, 2.2))
    x = np.zeros(n)
    for _ in range(kisi):
        f0 = rng.uniform(140, 280)
        hiz = rng.uniform(4.5, 6.5)
        t = zaman(n)
        darbe = np.clip(np.sin(2 * np.pi * hiz * t), 0, 1) ** 2 * np.exp(-t * rng.uniform(0.5, 1.5))
        ses = signal.sawtooth(2 * np.pi * np.cumsum(f0 * (1 - 0.15 * t / t[-1])) / SR) * darbe
        ses += filtre(gurultu(n), "bandpass", [800, 3000]) * darbe * 0.3
        x += sum(filtre(ses, "bandpass", [fm * 0.85, fm * 1.15], 2) * g for fm, g in [(750, 1), (1200, 0.6), (2500, 0.2)])
    return x


def poset():
    n = n_(rng.uniform(0.6, 1.5))
    x = np.zeros(n)
    for p in rng.integers(0, n, int(n / SR * 400)):
        x[p] = rng.uniform(-1, 1)
    return filtre(x, "highpass", 2500) * np.hanning(n)


def sayfa_cevir():
    n = n_(0.4)
    return filtre(gurultu(n), "bandpass", [1500, 7000]) * np.hanning(n) ** 1.5 * (1 + np.sin(2 * np.pi * 7 * zaman(n)))


def kalem():
    n = n_(rng.uniform(1, 2.5))
    gate = (filtre(rng.standard_normal(n), "lowpass", 8) > 0).astype(float)
    return filtre(gurultu(n), "bandpass", [2500, 8000]) * filtre(gate, "lowpass", 60, 2) * 0.4


def oksuruk():
    y = np.zeros(n_(1.2))
    for i in range(rng.integers(1, 3)):
        m = n_(0.25)
        s = filtre(gurultu(m), "bandpass", [300, 2500]) * zarf_ad(m, 0.01, 6)
        ekle(y, s, n_(i * 0.35))
    return y


def klavye():
    y = np.zeros(n_(4))
    t = 0.0
    while t < 3.5:
        m = n_(0.025)
        ekle(y, filtre(gurultu(m), "bandpass", [1500, 6000]) * np.exp(-np.linspace(0, 7, m)) * rng.uniform(0.3, 0.8), n_(t))
        t += rng.uniform(0.06, 0.2) + (rng.uniform(0.3, 0.6) if rng.uniform() < 0.1 else 0)
    return y


def telefon():
    y = np.zeros(n_(5))
    for r in range(2):
        m = n_(1.2)
        t = zaman(m)
        s = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) * (np.sin(2 * np.pi * 20 * t) > 0) * 0.3
        ekle(y, s, n_(r * 2.5))
    return y


def yazici():
    n = n_(4)
    t = zaman(n)
    x = filtre(gurultu(n), "bandpass", [300, 2000]) * (0.6 + 0.4 * (np.sin(2 * np.pi * 3 * t) > 0)) * 0.5
    return x * np.hanning(n) ** 0.3


def makas():
    y = np.zeros(n_(1.5))
    for i in range(rng.integers(3, 7)):
        m = n_(0.04)
        ekle(y, filtre(gurultu(m), "highpass", 3000) * np.exp(-np.linspace(0, 6, m)) * 0.7, n_(i * 0.22))
    return y


def tiras_makinesi():
    n = n_(rng.uniform(2, 4))
    t = zaman(n)
    s = signal.sawtooth(2 * np.pi * 120 * t) * 0.3 + filtre(gurultu(n), "bandpass", [2000, 6000]) * 0.3
    return s * np.hanning(n) ** 0.2 * (0.8 + 0.2 * np.sin(2 * np.pi * 1.5 * t))


def sprey():
    y = np.zeros(n_(1.2))
    for i in range(rng.integers(1, 4)):
        m = n_(0.2)
        ekle(y, filtre(gurultu(m), "highpass", 3500) * zarf_ad(m, 0.01, 3) * 0.6, n_(i * 0.3))
    return y


def cay_bardagi():
    y = tink(rng.uniform(2600, 3400), 0.4, 22) * 0.5
    z = np.zeros(n_(1.2))
    ekle(z, y, 0)
    ekle(z, kasik_karistirma()[: n_(1)] * 0.5, n_(0.2))
    return z


def seyyar(metin):
    return filtre(say_ses(metin, "Yelda", 165), "lowpass", 3500)


def ruzgar_uluma():
    n = n_(rng.uniform(4, 7))
    t = zaman(n)
    f = rng.uniform(350, 600) * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t))
    x = gurultu(n, "pembe")
    out = np.zeros(n)
    blok = n_(0.05)
    for b in range(0, n, blok):
        fc = f[b]
        out[b:b + blok] = filtre(x[max(0, b - n_(0.2)):b + blok], "bandpass", [fc * 0.95, fc * 1.05], 2)[-len(out[b:b + blok]):]
    return out * np.hanning(n) * 3


def kopek():
    y = np.zeros(n_(2))
    for i in range(rng.integers(1, 4)):
        m = n_(0.18)
        t = zaman(m)
        f0 = rng.uniform(350, 550)
        s = signal.sawtooth(2 * np.pi * np.cumsum(f0 * (1 - 0.4 * t / t[-1])) / SR) * zarf_ad(m, 0.01, 4)
        s = filtre(s, "bandpass", [500, 1800], 2) + filtre(gurultu(m), "bandpass", [600, 2000]) * zarf_ad(m, 0.005, 6) * 0.3
        ekle(y, s, n_(i * rng.uniform(0.3, 0.45)))
    return y


def baykus():
    y = np.zeros(n_(2.5))
    for i, (f, d) in enumerate([(420, 0.35), (380, 0.25), (400, 0.6)][: rng.integers(2, 4)]):
        m = n_(d)
        ekle(y, np.sin(2 * np.pi * f * zaman(m)) * np.hanning(m), n_(i * 0.55))
    return y


def kutuk():
    y = np.zeros(n_(1.5))
    ekle(y, filtre(gurultu(n_(0.3)), "lowpass", 400) * np.exp(-np.linspace(0, 5, n_(0.3))) * 0.6, 0)
    for _ in range(20):
        m = n_(0.006)
        ekle(y, filtre(rng.standard_normal(m), "highpass", 2000, 2) * rng.uniform(0.2, 1), n_(rng.exponential(0.3)))
    return y


def tibet_kasesi():
    f = rng.uniform(180, 320)
    return kismi([f, f * 1.004, f * 2.71, f * 5.12], [0.5, 0.5, 0.9, 1.6], 6, [1, 0.8, 0.4, 0.15]) * zarf_ad(n_(6), 0.05, 0)


def ruzgar_cani():
    y = np.zeros(n_(4))
    penta = [72, 74, 76, 79, 81, 84]
    for i in range(rng.integers(3, 8)):
        f = midi(random.choice(penta))
        ekle(y, kismi([f, f * 2.76], [1.5, 3], 2.5, [1, 0.3]) * rng.uniform(0.3, 0.8), n_(rng.uniform(0, 1.5)))
    return y


def bip():
    y = np.zeros(n_(1.5))
    f = random.choice([880, 1046, 1318, 1568])
    for i in range(rng.integers(1, 4)):
        m = n_(0.09)
        ekle(y, np.sin(2 * np.pi * f * zaman(m)) * np.hanning(m) ** 0.2 * 0.5, n_(i * 0.18))
    return y


def hava_kilidi():
    n = n_(3)
    return filtre(gurultu(n), "highpass", 1500) * zarf_ad(n, 0.05, 2) * 0.6


def telsiz(metin):
    s = filtre(say_ses(metin, random.choice(["Daniel", "Samantha", "Yelda"]), 190), "bandpass", [500, 2800])
    k = filtre(gurultu(n_(0.15)), "highpass", 1000) * 0.2
    return np.concatenate([k, s * 0.8 + filtre(gurultu(len(s)), "highpass", 2000) * 0.03, k])


def servis_arabasi():
    n = n_(5)
    x = filtre(gurultu(n), "bandpass", [150, 800]) * 0.3
    for _ in range(25):
        ekle(x, tink(rng.uniform(2000, 3500), 0.15, 40) * 0.15, int(rng.uniform(0, n - n_(0.2))))
    return x * np.hanning(n)


def halat():
    y = np.zeros(n_(2))
    for i in range(rng.integers(2, 5)):
        f = rng.uniform(600, 900)
        ekle(y, kismi([f, f * 2.3], [8, 12], 0.5, [1, 0.5]) * rng.uniform(0.4, 0.9), n_(i * rng.uniform(0.3, 0.5)))
    return y


def tekne_gecisi():
    n = n_(8)
    t = zaman(n)
    s = sum(np.sin(2 * np.pi * 70 * k * t) / k for k in range(1, 5)) * 0.4 + filtre(gurultu(n, "pembe"), "lowpass", 900)
    return s * np.exp(-((t - 4) / 2) ** 2)


def grup_soyle(metin, kisi=6, hiz=190):
    tek = say_ses(metin, "Yelda", hiz)
    x = np.zeros(len(tek) + n_(0.5))
    for _ in range(kisi):
        ekle(x, tek * rng.uniform(0.5, 1), n_(rng.uniform(0, 0.15)))
    return filtre(x, "lowpass", 3000)


# ---------- ortam tanımları ----------
# ort: ortalama aralık (sn), min: en kısa aralık, kazanc: göreli yükseklik, yuru: soldan sağa geçer
# sabit: hızı/perdesi değiştirilmez (konuşma, müzik), yanki: sn (oda), uzak: lowpass (Hz)
def O(ad, uret, adet, ort, min_, kazanc, **kw):
    return dict(ad=ad, uret=uret, adet=adet, ort=ort, min=min_, kazanc=kazanc, **kw)


def ortamlar():
    R = lambda *a: random.choice(a)
    kafe_masa = lambda: yakin_masa(R(*(CUMLELER * 2 + YABANCI)), R("Yelda", "Yelda", "Samantha", "Daniel"))
    sessiz_masa = lambda: filtre(yakin_masa(R(*CUMLELER), R("Yelda", "Samantha")), "lowpass", 1800) * 0.6
    return {
        # ---- Zengin ----
        "restoran": dict(baslik="Lüks restoran", alt="piyano, kadeh", grup="Zengin",
            yataklar=[lambda n: piyano_yatak(n, 0) * 0.5 + oda_tonu(n, 200, 0.03),
                      lambda n: stereo_yanki(ugultu(n, 10, (150, 170), 1200), 1.2, 0.5) * 0.6 + oda_tonu(n, 200, 0.03)],
            sureler=(64, 48), olaylar=[
                O("catal", catal_bicak, 8, 5, 1.5, 0.3, yanki=0.8),
                O("kadeh", kadeh, 4, 35, 12, 0.35, yanki=1.0),
                O("sampanya", sampanya, 3, 120, 50, 0.45, yanki=1.0),
                O("sarap", sarap_dokme, 3, 45, 20, 0.3, yanki=0.6),
                O("garson", lambda: garson(R("Şefin bugünkü önerisi", "Şarabınızı tazeleyeyim mi?", "Afiyet olsun efendim", None)),
                  5, 45, 20, 0.5, yuru=True, sabit=True, yanki=0.8),
                O("masa", sessiz_masa, 6, 25, 8, 0.3, sabit=True, yanki=0.8),
                O("gulme", lambda: gulusme(2), 4, 60, 25, 0.15, yanki=1.0, uzak=2500),
            ]),
        "lobi": dict(baslik="Otel lobisi", alt="mermer, piyano", grup="Zengin",
            yataklar=[lambda n: piyano_yatak(n, 3) * 0.35 + oda_tonu(n, 250, 0.04),
                      lambda n: stereo_yanki(ugultu(n, 6, (160, 180), 1300), 2.2, 0.6) * 0.5],
            sureler=(64, 48), olaylar=[
                O("adim", lambda: adimlar(rng.integers(6, 12), rng.uniform(0.45, 0.6), 2.0), 6, 12, 4, 0.4, yuru=True, yanki=2.2),
                O("bavul", bavul, 3, 40, 15, 0.4, yuru=True, yanki=2.0),
                O("asansor", asansor, 3, 50, 20, 0.35, yanki=2.0),
                O("zil", resepsiyon_zili, 2, 60, 25, 0.3, yanki=2.0),
                O("karsilama", lambda: yakin_masa(R("Hoş geldiniz efendim", "Odanız hazır", "Bavullarınızı alayım", "İyi akşamlar"), "Yelda"),
                  4, 50, 20, 0.35, sabit=True, yanki=1.8),
            ]),
        "yat": dict(baslik="Yat", alt="deniz, buz", grup="Zengin",
            yataklar=[lambda n: su_sapirti(n) * 0.6 + oda_tonu(n, 400, 0.08),
                      lambda n: yatak_sahil(n) * 0.4],
            olaylar=[
                O("halat", halat, 4, 15, 5, 0.25, yanki=0.5),
                O("buz", buz_bardak, 4, 30, 10, 0.4),
                O("sampanya", sampanya, 2, 150, 60, 0.4),
                O("marti", marti, 4, 35, 12, 0.2, yanki=1.2, uzak=5000),
                O("tekne", tekne_gecisi, 2, 90, 40, 0.3, yuru=True, uzak=2000),
                O("savrulma", lambda: ruzgar_savrulma(rng.uniform(4, 7)), 3, 30, 12, 0.3),
            ]),
        "kumarhane": dict(baslik="Kumarhane", alt="slot, rulet", grup="Zengin",
            yataklar=[lambda n: stereo_yanki(ugultu(n, 14, (170, 200), 1400), 1.5, 0.5),
                      lambda n: oda_tonu(n, 300, 0.08) + stereo_yanki(ugultu(n, 8, (180, 210), 1200), 1.5, 0.5) * 0.5],
            olaylar=[
                O("slot", slot, 8, 4, 1, 0.2, sabit=True, yanki=1.0),
                O("rulet", rulet, 3, 30, 15, 0.35, yanki=0.8),
                O("fis", fis, 6, 7, 2, 0.3, yanki=0.6),
                O("kadeh", buz_bardak, 3, 25, 10, 0.25, yanki=0.8),
                O("tezahurat", lambda: grup_soyle(R("Evet!", "Kazandık!", "Bir daha!"), 5), 3, 60, 25, 0.4, sabit=True, yanki=1.5),
                O("gulme", gulusme, 4, 35, 12, 0.25, yanki=1.2, uzak=3000),
            ]),
        "jet": dict(baslik="Özel jet", alt="sessiz kabin", grup="Zengin",
            yataklar=[lambda n: motor_yatak(n, 110, 500, 0.6), lambda n: motor_yatak(n, 220, 1200, 0.3) * 0.5],
            olaylar=[
                O("kadeh", buz_bardak, 3, 40, 15, 0.3),
                O("sampanya", sampanya, 2, 180, 80, 0.35),
                O("kemer", lambda: kismi([880], [2], 1.5) * 0.4, 1, 150, 60, 0.25),
                O("anons", lambda: anons(R("Sayın misafirlerimiz, inişe yaklaşık kırk dakika var",
                                           "Kaptanınız konuşuyor, hava sakin, keyfinize bakın")), 2, 240, 120, 0.35, sabit=True),
            ]),
        # ---- Şehir ----
        "kafe": dict(baslik="Kafe", alt="garson, fincan", grup="Şehir",
            yataklar=[yatak_kafe, lambda n: yatak_kafe(n) * 0.8], olaylar=[
                O("fincan", lambda: tink(rng.uniform(1800, 3200), 0.5, 18) * 0.6, 8, 6, 1.5, 0.35, yanki=0.6),
                O("tabak", fincan_tabak, 5, 12, 3, 0.4, yanki=0.6),
                O("kasik", kasik_karistirma, 4, 25, 8, 0.3, yanki=0.5),
                O("sandalye", sandalye, 6, 22, 6, 0.35, yanki=0.7),
                O("garson", lambda: garson(R("Buyrun, afiyet olsun", "Hemen geliyorum", "Çaylar geldi", "Hesabınız",
                                             "Başka bir isteğiniz var mı?", None, None)), 6, 40, 15, 0.7, yuru=True, sabit=True, yanki=0.6),
                O("devirme", devirme, 3, 200, 90, 0.9, sabit=True, yanki=0.9),
                O("espresso", espresso, 3, 55, 25, 0.3, yanki=0.7),
                O("kapi", kapi_zili, 3, 90, 35, 0.35, yanki=0.8),
                O("masa", kafe_masa, 8, 18, 6, 0.35, sabit=True, yanki=0.7),
                O("gulme", gulusme, 3, 70, 25, 0.2, yanki=0.8, uzak=3000),
            ]),
        "meyhane": dict(baslik="Meyhane", alt="şerefe, tabak", grup="Şehir",
            yataklar=[lambda n: stereo_yanki(ugultu(n, 14, (170, 195), 1500), 1.0, 0.45),
                      lambda n: stereo_yanki(ugultu(n, 10, (175, 200), 1400), 1.0, 0.45) * 0.8],
            olaylar=[
                O("kadeh", kadeh, 5, 15, 4, 0.35, yanki=0.7),
                O("serefe", lambda: grup_soyle(R("Şerefe!", "Şerefinize!", "Hadi bakalım, şerefe!"), rng.integers(3, 7)), 4, 40, 15, 0.5, sabit=True, yanki=0.8),
                O("tabak", fincan_tabak, 5, 10, 3, 0.4, yanki=0.6),
                O("catal", catal_bicak, 6, 7, 2, 0.3, yanki=0.6),
                O("garson", lambda: garson(R("Mezeler geldi", "Bir duble daha mı?", "Kalamar sıcak sıcak", None)), 4, 50, 20, 0.6, yuru=True, sabit=True, yanki=0.6),
                O("gulme", gulusme, 5, 20, 7, 0.35, yanki=0.8, uzak=3500),
                O("masa", kafe_masa, 6, 15, 5, 0.35, sabit=True, yanki=0.7),
                O("devirme", devirme, 2, 300, 150, 0.8, sabit=True, yanki=0.9),
            ]),
        "pazar": dict(baslik="Semt pazarı", alt="bağıran satıcılar", grup="Şehir",
            yataklar=[lambda n: ugultu(n, 16, (180, 210), 2000) * 0.9 + oda_tonu(n, 500, 0.05),
                      lambda n: ugultu(n, 12, (180, 210), 1800) * 0.8],
            olaylar=[
                O("satici", lambda: seyyar(R("Taze domates, gel abla gel!", "Bal gibi kavun burada!", "Hepsi on lira, hepsi on lira!",
                                             "Mandalina tatlı mı tatlı!", "Gel gel, tadına bak!", "Yerli fasulye!")),
                  8, 9, 3, 0.55, sabit=True, yanki=1.0),
                O("poset", poset, 5, 8, 2, 0.3),
                O("terazi", lambda: kismi([620, 1500], [8, 15], 0.5, [1, 0.4]) * 0.6, 3, 15, 5, 0.3),
                O("adim", lambda: adimlar(8, 0.5, 0.6), 3, 10, 4, 0.2, yuru=True),
            ]),
        "berber": dict(baslik="Berber", alt="makas, çay", grup="Şehir",
            yataklar=[lambda n: stereo_yanki(ugultu(n, 3, (170, 190), 1600), 0.8, 0.4) * 0.5 + oda_tonu(n, 300, 0.05),
                      lambda n: oda_tonu(n, 300, 0.06)],
            olaylar=[
                O("makas", makas, 6, 5, 1.5, 0.45, yanki=0.4),
                O("tiras", tiras_makinesi, 3, 35, 15, 0.3, yanki=0.4),
                O("sprey", sprey, 3, 30, 12, 0.35, yanki=0.4),
                O("cay", cay_bardagi, 3, 40, 15, 0.35, yanki=0.5),
                O("usta", lambda: yakin_masa(R("Kolay gelsin", "Enseyi biraz daha alalım mı?", "Bir çay söyle oğlum",
                                               "Ayna şöyle, beğendin mi?", "Maç kaç kaç bitti dün?"), "Yelda"), 5, 30, 12, 0.45, sabit=True, yanki=0.5),
                O("kapi", kapi_zili, 2, 120, 50, 0.3, yanki=0.6),
            ]),
        "kulup": dict(baslik="Gece kulübü", alt="bas, kalabalık", grup="Şehir",
            yataklar=[kulup_yatak, lambda n: stereo_yanki(ugultu(n, 10, (200, 230), 1200), 1.2, 0.5) * 0.4],
            sureler=(48, 64), olaylar=[
                O("tezahurat", lambda: grup_soyle(R("Hey!", "Oley!", "Woo!"), 8), 3, 30, 12, 0.35, sabit=True, yanki=1.2),
                O("kadeh", buz_bardak, 4, 15, 5, 0.25, yanki=0.8),
                O("gulme", gulusme, 3, 25, 10, 0.25, yanki=1.0, uzak=3000),
                O("bagiran", lambda: yakin_masa(R("Şarkı efsane!", "Bir şey içelim mi?", "Ne? Duyamıyorum!"), "Yelda"), 3, 35, 15, 0.4, sabit=True, yanki=0.8),
            ]),
        "tribun": dict(baslik="Tribün", alt="maç havası", grup="Şehir",
            yataklar=[yatak_tribun, lambda n: yatak_tribun(n) * 0.9], olaylar=[
                O("tezahurat", lambda: tezahurat(R("Haydi! Haydi! Haydi!", "Gol istiyoruz! Gol istiyoruz!", "Ne mutlu bize!",
                                                   "Bas bas bas!", "Hep beraber!")), 5, 28, 12, 0.6, sabit=True, yanki=2.2),
                O("davul", lambda: davul(R([1, 0, 0, 1, 0, 0, 1, 0] * 3, [1, 0, 1, 0, 1, 1, 1, 0] * 3, [1, 1, 1, 0] * 6)), 3, 35, 15, 0.5, yanki=1.8),
                O("alkis", lambda: alkis(rng.uniform(1.5, 3.5)), 4, 20, 6, 0.35, yanki=1.5),
                O("duduk", hakem_duduk, 4, 30, 10, 0.2, yanki=2.0, uzak=4500),
                O("yuhalama", islik_yuhalama, 3, 80, 40, 0.45, yanki=2.0),
                O("bagiran", lambda: yakin_masa(R("Hakem!", "Vur vur!", "Hadi be!", "Ne yapıyorsun ya!", "Pas ver pas!", "Hadi oğlum!"), "Yelda"),
                  6, 22, 8, 0.45, sabit=True, yanki=1.0),
                O("gol", gol, 2, 240, 120, 1.0, yanki=2.2),
                O("korna", korna, 3, 60, 20, 0.2, yanki=2.0),
            ]),
        # ---- Yolculuk ----
        "vapur": dict(baslik="Vapur", alt="martı, simit, çay", grup="Yolculuk",
            yataklar=[lambda n: motor_yatak(n, 55, 400, 0.7) + su_sapirti(n) * 0.3,
                      lambda n: stereo_yanki(ugultu(n, 8, (170, 190), 1400), 1.0, 0.4) * 0.4],
            olaylar=[
                O("marti", marti, 5, 15, 5, 0.3, yanki=1.2, uzak=5000),
                O("simitci", lambda: seyyar(R("Simit, taze simit!", "Çay, çay isteyen?", "Sıcak simit!")), 4, 45, 20, 0.45, yuru=True, sabit=True, yanki=0.8),
                O("cay", cay_bardagi, 4, 20, 6, 0.35, yanki=0.5),
                O("duduk", vapur, 2, 180, 80, 0.35, yanki=2.5, uzak=900),
                O("masa", kafe_masa, 5, 25, 8, 0.3, sabit=True, yanki=0.6),
            ]),
        "tren": dict(baslik="Tren", alt="ray, anons", grup="Yolculuk",
            yataklar=[ray_yatak, lambda n: motor_yatak(n, 48, 400, 0.4)], sureler=(44, 66), olaylar=[
                O("anons", lambda: anons(R("Sayın yolcularımız, bir sonraki durak Eskişehir", "Yemekli vagonumuz hizmetinizdedir",
                                           "Lütfen eşyalarınızı yanınızda bulundurunuz")), 3, 150, 70, 0.35, sabit=True),
                O("duduk", lambda: korna() * 0.6, 2, 120, 50, 0.3, yanki=2.0, uzak=2000),
                O("servis", servis_arabasi, 2, 120, 50, 0.3, yuru=True),
                O("sayfa", sayfa_cevir, 3, 30, 10, 0.2),
                O("masa", sessiz_masa, 4, 30, 12, 0.3, sabit=True),
            ]),
        "uzay": dict(baslik="Uzay istasyonu", alt="bip, telsiz", grup="Yolculuk",
            yataklar=[uzay_yatak, lambda n: oda_tonu(n, 800, 0.1)], olaylar=[
                O("bip", bip, 5, 10, 3, 0.25, sabit=True, yanki=0.5),
                O("kilit", hava_kilidi, 2, 90, 40, 0.35, yanki=0.6),
                O("telsiz", lambda: telsiz(R("Houston, all systems nominal", "Kontrol, burası istasyon, her şey yolunda",
                                             "Copy that, docking in ten minutes")), 3, 60, 25, 0.35, sabit=True),
            ]),
        # ---- İç mekân ----
        "ofis": dict(baslik="Ofis", alt="klavye, telefon", grup="İç mekân",
            yataklar=[lambda n: oda_tonu(n, 400, 0.08) + stereo_yanki(ugultu(n, 5, (170, 190), 1400), 0.8, 0.4) * 0.35,
                      lambda n: oda_tonu(n, 250, 0.06)],
            olaylar=[
                O("klavye", klavye, 6, 8, 2, 0.3, yanki=0.4),
                O("telefon", telefon, 2, 90, 40, 0.25, yanki=0.6, uzak=3000),
                O("yazici", yazici, 2, 70, 30, 0.25, yanki=0.6),
                O("kahve", espresso, 2, 120, 50, 0.2, yanki=0.6),
                O("masa", sessiz_masa, 5, 30, 10, 0.3, sabit=True, yanki=0.6),
                O("sandalye", sandalye, 3, 40, 15, 0.25, yanki=0.6),
            ]),
        "kutuphane": dict(baslik="Kütüphane", alt="sayfa, fısıltı", grup="İç mekân",
            yataklar=[lambda n: oda_tonu(n, 200, 0.05), lambda n: oda_tonu(n, 400, 0.03)],
            olaylar=[
                O("sayfa", sayfa_cevir, 5, 8, 2, 0.35, yanki=1.2),
                O("kalem", kalem, 4, 20, 6, 0.25, yanki=1.0),
                O("adim", lambda: adimlar(8, 0.6, 1.5), 3, 40, 15, 0.25, yuru=True, yanki=1.8),
                O("oksuruk", oksuruk, 3, 70, 30, 0.3, yanki=1.5),
                O("fisilti", lambda: say_ses(R("Can you pass me that book", "Let's go get coffee after this", "I can't find the chapter"),
                                              "Whisper", 170) * 0.6, 3, 50, 20, 0.2, sabit=True, yanki=1.5),
                O("kitap", lambda: filtre(gurultu(n_(0.15)), "lowpass", 400) * np.exp(-np.linspace(0, 6, n_(0.15))), 3, 45, 15, 0.35, yanki=1.8),
            ]),
        "somine": dict(baslik="Şömine başı", alt="ateş, saat, rüzgâr", grup="İç mekân",
            yataklar=[lambda n: ates_yatak(n) + saat_tik(n), lambda n: oda_tonu(n, 150, 0.06)], sureler=(44, 66),
            olaylar=[
                O("kutuk", kutuk, 4, 30, 10, 0.4),
                O("uluma", ruzgar_uluma, 3, 40, 15, 0.25, uzak=1500),
                O("bardak", lambda: tink(rng.uniform(1500, 2200), 0.6, 10) * 0.6, 3, 50, 20, 0.25, yanki=0.5),
            ]),
        "spa": dict(baslik="Spa", alt="su, kase", grup="İç mekân",
            yataklar=[lambda n: su_sirildama(n), lambda n: oda_tonu(n, 300, 0.04)],
            olaylar=[
                O("kase", tibet_kasesi, 3, 40, 15, 0.35, sabit=True, yanki=2.0),
                O("can", ruzgar_cani, 3, 25, 10, 0.25, sabit=True, yanki=1.5),
            ]),
        # ---- Doğa ----
        "orman": dict(baslik="Orman", alt="kuş, rüzgâr", grup="Doğa",
            yataklar=[yatak_orman, lambda n: yatak_orman(n) * 0.8], olaylar=[
                O("kus", kus_dizisi, 10, 4, 0.8, 0.25, yanki=0.8),
                O("agackakan", agackakan, 3, 60, 25, 0.3, yanki=1.0),
                O("guguk", guguk, 2, 90, 40, 0.2, yanki=1.5, uzak=2500),
                O("dal", dal_kirilmasi, 3, 70, 30, 0.35, yanki=0.8),
                O("savrulma", lambda: ruzgar_savrulma(rng.uniform(4, 8)), 3, 30, 12, 0.4),
            ]),
        "sahil": dict(baslik="Sahil", alt="dalga, martı", grup="Doğa",
            yataklar=[yatak_sahil, lambda n: yatak_sahil(n) * 0.8], olaylar=[
                O("dalga", dalga, 8, 8, 4, 0.8),
                O("marti", marti, 4, 30, 10, 0.25, yanki=1.2, uzak=5000),
                O("vapur", vapur, 2, 150, 70, 0.25, yanki=2.5, uzak=700),
                O("savrulma", lambda: ruzgar_savrulma(rng.uniform(4, 7)), 3, 35, 15, 0.3),
            ]),
        "yagmur": dict(baslik="Yağmur", alt="cama vuran", grup="Doğa",
            yataklar=[yatak_yagmur, lambda n: yatak_yagmur(n) * 0.8], olaylar=[
                O("gok", lambda: gok_gurultusu(rng.uniform() < 0.3), 4, 70, 30, 0.8, yanki=1.5),
                O("araba", araba_gecisi, 4, 35, 12, 0.35, yuru=True, yanki=0.5),
                O("oluk", oluk_damla, 3, 45, 20, 0.3, yanki=0.4),
                O("savrulma", lambda: ruzgar_savrulma(rng.uniform(4, 7)), 3, 40, 15, 0.35),
            ]),
        "firtina": dict(baslik="Fırtına", alt="gök, uluyan rüzgâr", grup="Doğa",
            yataklar=[lambda n: yatak_yagmur(n) * 1.2 + oda_tonu(n, 120, 0.1), lambda n: yatak_yagmur(n)],
            olaylar=[
                O("gok", lambda: gok_gurultusu(rng.uniform() < 0.5), 5, 25, 10, 1.0, yanki=1.5),
                O("uluma", ruzgar_uluma, 4, 15, 5, 0.4),
                O("savrulma", lambda: ruzgar_savrulma(rng.uniform(4, 7)), 3, 15, 6, 0.5),
            ]),
        "kamp": dict(baslik="Kamp ateşi", alt="çıtırtı, baykuş", grup="Doğa",
            yataklar=[lambda n: ates_yatak(n), lambda n: cekirge_yatak(n, 5)], olaylar=[
                O("kutuk", kutuk, 4, 20, 7, 0.45),
                O("baykus", baykus, 3, 60, 25, 0.25, sabit=True, yanki=1.8, uzak=1500),
                O("dal", dal_kirilmasi, 3, 60, 25, 0.25, yanki=1.0, uzak=4000),
                O("kopek", kopek, 2, 150, 70, 0.15, yanki=2.0, uzak=1500),
            ]),
        "gece": dict(baslik="Yaz gecesi", alt="cırcır, köpek", grup="Doğa",
            yataklar=[lambda n: cekirge_yatak(n, 8), lambda n: oda_tonu(n, 300, 0.04) + yatak_orman(n) * 0.2],
            olaylar=[
                O("kopek", kopek, 4, 50, 20, 0.2, yanki=2.0, uzak=1800),
                O("araba", araba_gecisi, 3, 70, 30, 0.15, yuru=True, uzak=1500),
                O("baykus", baykus, 2, 90, 40, 0.15, sabit=True, yanki=1.8, uzak=1500),
                O("savrulma", lambda: ruzgar_savrulma(rng.uniform(4, 8)), 3, 40, 15, 0.25),
            ]),
    }


def mp3_yaz(yol, x, kbps):
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "x.wav")
        wavfile.write(wav, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-b:a", f"{kbps}k", yol], check=True)


def ortam_uret(ad, tanim):
    klasor = os.path.join(CIKTI, ad)
    if os.path.isdir(klasor):
        shutil.rmtree(klasor)
    os.makedirs(klasor)
    kayit = {k: tanim[k] for k in ("baslik", "alt", "grup")} | {"yatak": [], "olaylar": []}
    for i, (yatak, sure) in enumerate(zip(tanim["yataklar"], tanim.get("sureler", YATAK_SURELERI))):
        st = dongu_yap(yatak(n_(sure + BINDIRME)), sure)
        st = st / (np.sqrt(np.mean(st ** 2)) + 1e-9) * (0.1 if i == 0 else 0.06)
        st = st / max(1, np.max(np.abs(st)) / 0.95)
        dosya = f"yatak{i + 1}.mp3"
        mp3_yaz(os.path.join(klasor, dosya), st, 96)
        kayit["yatak"].append(f"{ad}/{dosya}")
    for o in tanim["olaylar"]:
        dosyalar = []
        for k in range(o["adet"]):
            x = np.asarray(o["uret"](), dtype=float)
            if o.get("uzak"):
                x = filtre(x, "lowpass", o["uzak"])
            if o.get("yanki"):
                x = yanki(x, o["yanki"], 0.35)
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
            bit = np.nonzero(np.abs(x) > 0.002)[0]
            x = x[: (bit[-1] + n_(0.05)) if len(bit) else len(x)]
            x[-n_(0.05):] *= np.linspace(1, 0, n_(0.05))
            dosya = f"{o['ad']}{k + 1}.mp3"
            mp3_yaz(os.path.join(klasor, dosya), x, 80)
            dosyalar.append(f"{ad}/{dosya}")
        kayit["olaylar"].append({k: o[k] for k in ("ad", "ort", "min", "kazanc")} |
                                {"dosyalar": dosyalar, "yuru": bool(o.get("yuru")), "sabit": bool(o.get("sabit"))})
    return kayit


def main():
    import sys
    secili = sys.argv[1:]  # boşsa hepsi; ad verilirse sadece onlar yeniden üretilir
    yol = os.path.join(CIKTI, "manifest.json")
    os.makedirs(CIKTI, exist_ok=True)
    manifest = json.load(open(yol, encoding="utf-8")) if os.path.exists(yol) and secili else {}
    tanimlar = ortamlar()
    for ad, tanim in tanimlar.items():
        if secili and ad not in secili:
            continue
        manifest[ad] = ortam_uret(ad, tanim)
        print(ad, "tamam:", sum(len(o["dosyalar"]) for o in manifest[ad]["olaylar"]), "olay sesi", flush=True)
    manifest = {ad: manifest[ad] for ad in tanimlar if ad in manifest}  # sıra tanımdaki gibi
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
