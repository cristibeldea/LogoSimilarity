import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
import subprocess
from joblib import Parallel, delayed
import imagehash
from PIL import Image as PILImage

INPUT_FOLDER = 'logos' #logo-uri neclasificate
PROCESSED_FOLDER = 'processed_bin' #folder cu conturul imaginilor pentru debug
OUTPUT_FOLDER = 'grouped_logos' #folderul de output, cu clusterele formate
IMG_SIZE = 128 #normalizam la 128x128
#THRESHOLD = 0.35
THRESHOLD = 0.39 #determinat experimental. in functie de dorinta strictetii filtrului puteti ajusta atat aceasta variabila
                 #cat si HASH_THRESHOLD
HASH_THRESHOLD = 16
MAX_IMAGES = None #Modificati la un numar mai mic pentru a rula programul pe un batch mai mic de poza (pentru test rapid)

def svg_to_png_rsvg(svg_path, output_path): #Converteste imaginile din format .svg in format .png
    try:
        subprocess.run([
            "rsvg-convert",
            "-w", str(IMG_SIZE),
            "-h", str(IMG_SIZE),
            "-f", "png",
            "-o", output_path,
            svg_path
        ], check=True)
        return True
    except Exception as e:
        print(f"[ERROR rsvg-convert] {svg_path}: {e}")
        return False

def compute_hash(img):
    #folosim imagehash (cu differnece hash, dhash) ca prim filtru rapid de verificare a similaritatii pozelor
    return imagehash.dhash(PILImage.fromarray(img).convert('L'))

def process_image(file):
    #Returneaza:
    # imaginea RGB redimensionată
    # versiunea binarizată pentru analiza shape-ului
    # hash vizual (dhash) pentru comparație rapidă la inceput
    # aria formei din imagine
    # un mesaj de stare ([OK], [SKIP], [ERROR])


    name, ext = os.path.splitext(file.lower())
    ext = ext.strip('.')
    supported = ['png', 'jpg', 'jpeg', 'svg', 'ico']
    if ext not in supported:
        return file, None, None, None, f"[SKIP] {file}: format not supported"

    input_path = os.path.join(INPUT_FOLDER, file)
    try:
        if ext == 'svg':
            output_temp = os.path.join(PROCESSED_FOLDER, f"{name}_temp.png")
            if not svg_to_png_rsvg(input_path, output_temp):
                return file, None, None, None, f"[FAIL SVG] {file}"
            img = Image.open(output_temp).convert("RGB")
        else:
            img = Image.open(input_path).convert("RGB")

        rgb_img = np.array(img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR))
        img_hash = compute_hash(rgb_img)

        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        var_map = cv2.Laplacian(gray, cv2.CV_64F)
        var_map = np.absolute(var_map)
        var_max = var_map.max()
        if var_max == 0:
            return file, None, None, None, f"[SKIP] {file}: uniform image"
        var_map = (var_map / var_max * 255).astype(np.uint8)
        _, mask = cv2.threshold(var_map, 10, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return file, None, None, None, f"[SKIP] {file}: no content"

        x, y, w, h = cv2.boundingRect(coords)
        pad = int(max(w, h) * 0.15)
        x, y = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + 2 * pad, gray.shape[1]), min(y + h + 2 * pad, gray.shape[0])

        cropped = gray[y:y2, x:x2]
        resized = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_AREA)
        _, bin_img = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_img = cv2.bitwise_not(bin_img)
        area = cv2.countNonZero(bin_img)

        output_path = os.path.join(PROCESSED_FOLDER, f"{name}.png")
        cv2.imwrite(output_path, bin_img)

        return file, area, rgb_img, img_hash, f"[OK] {file}"

    except Exception as e:
        return file, None, None, None, f"[ERROR] {file}: {str(e)}"

def parallel_process_images(files): #Pentru eficienta aflam distantele pe fiecare filtru folosind paralelizarea
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    results = Parallel(n_jobs=-1)(delayed(process_image)(f) for f in tqdm(files, desc="Procesare imagini"))
    for _, _, _, _, msg in results:
        print(msg)
    areas = {f: a for f, a, _, _, _ in results if a is not None}
    color_images = {f: c for f, _, c, _, _ in results if c is not None}
    hashes = {f: h for f, _, _, h, _ in results if h is not None}
    return areas, color_images, hashes

def load_shape_images(areas): #folosim imaginile salvate in 'processed_bin' sub forma de shapes
    files = [f for f in os.listdir(PROCESSED_FOLDER) if f.lower().endswith('.png') and f in areas]
    return {f: cv2.imread(os.path.join(PROCESSED_FOLDER, f), cv2.IMREAD_GRAYSCALE) for f in files}

def hsv_color_similarity(img1, img2):
    # Returnează un scor de similaritate intre doua imagini in functie de media culorilor in HSV
    # Ignora albul sau culorile apropiate de alb (deoarece mare parte din fundaluri sunt albe)
    def get_hsv_mean(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = hsv[:, :, 2] < 245  # exclude white
        h_mean = np.mean(hsv[:, :, 0][mask]) if np.any(mask) else 0
        s_mean = np.mean(hsv[:, :, 1][mask]) if np.any(mask) else 0
        v_mean = np.mean(hsv[:, :, 2][mask]) if np.any(mask) else 0
        return np.array([h_mean, s_mean, v_mean])

    m1 = get_hsv_mean(img1)
    m2 = get_hsv_mean(img2)
    dist = np.linalg.norm(m1 - m2)
    return min(dist / 255, 1.0)

def shape_similarity(img1, img2, a1, a2):
    # Calculează diferența de formă dintre două imagini binarizate (alb-negru).
    # Normalizează dimensiunile imaginilor pe baza ariei pentru comparație echitabilă.
    # Creează un canvas comun, inversează imaginile și compară pixelii folosind XOR.
    # Returnează un scor între 0 și 1, unde 0 înseamnă forme identice, iar 1 complet diferite.

    if a1 > a2 and a2 > 0:
        scale = np.sqrt(a2 / a1)
        img1 = cv2.resize(img1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif a2 > a1 and a1 > 0:
        scale = np.sqrt(a1 / a2)
        img2 = cv2.resize(img2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])
    c1 = np.full((h, w), 255, dtype=np.uint8)
    c2 = np.full((h, w), 255, dtype=np.uint8)
    c1[:img1.shape[0], :img1.shape[1]] = img1
    c2[:img2.shape[0], :img2.shape[1]] = img2

    inv1 = cv2.bitwise_not(c1)
    inv2 = cv2.bitwise_not(c2)
    diff = cv2.bitwise_xor(inv1, inv2)
    return np.count_nonzero(diff) / max(np.count_nonzero(cv2.bitwise_or(inv1, inv2)), 1)


def compare_pair(f1, shape1, a1, f2, shape2, a2, color_imgs): #face suma ponderata intre rezultatul filtrului de shape si filtrului de culoare
    color_score = hsv_color_similarity(color_imgs[f1], color_imgs[f2])
    shape_score = shape_similarity(shape1, shape2, a1, a2)
    #return (f1, f2, 0.78 * shape_score + 0.22 * color_score)
    return (f1, f2, 0.7 * shape_score + 0.3 * color_score) #ponderile sunt alese experimental. Mai multe configuratii pot functiona

def build_similarity_matrix(shape_imgs, areas, color_imgs, hashes): #creeaza o matrice de similaritate cu distantele finale dintre fiecare 2 poze
    files = list(shape_imgs.items())
    enriched = [(name, img, areas[name]) for name, img in files]
    tasks = [
        (e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], color_imgs)
        for i, e1 in enumerate(enriched) for j, e2 in enumerate(enriched) if j > i
        if (hashes[e1[0]] - hashes[e2[0]]) <= HASH_THRESHOLD
    ]
    print(f"Se compară {len(tasks)} perechi din {len(files)} imagini")
    return Parallel(n_jobs=-1, batch_size=20)(delayed(compare_pair)(*task) for task in tqdm(tasks, desc="Comparare"))

def group_from_similarity(sim_matrix, files):
    #in functie de distantele gasite, algoritmul grupeaza pozele in acelasi cluster daca au distanta intre ele mai mica decat
    #pragul ales. Odata o poza repartizata intr-un cluster, ea nu mai este luata in considerare pentru un alt cluster,
    #in felul acesta, evitand clusterizare in lant. (de genul A seamana cu B, B seamana cu C, deci A seamana cu C).
    groups = []
    grouped = set()
    neighbors = {f: set() for f in files}
    for f1, f2, score in sim_matrix:
        if score <= THRESHOLD:
            neighbors[f1].add(f2)
            neighbors[f2].add(f1)
    for f in files:
        if f in grouped:
            continue
        queue = [f]
        group = set()
        while queue:
            current = queue.pop()
            if current in grouped:
                continue
            grouped.add(current)
            group.add(current)
            queue.extend(neighbors[current] - grouped)
        groups.append(list(group))
    return groups

def save_groups(groups): # salvam clusterele in foldere separate
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for i, group in enumerate(groups):
        group_dir = os.path.join(OUTPUT_FOLDER, f"group_{i+1}")
        os.makedirs(group_dir, exist_ok=True)
        for f in group:
            input_path = os.path.join(INPUT_FOLDER, f)
            if os.path.exists(input_path):
                shutil.copy(input_path, os.path.join(group_dir, f))

if __name__ == "__main__":
    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.ico'))]
    files_to_process = sorted(all_files)[:MAX_IMAGES] #am adaugat un prag maxim de imagini in cazul in care se doreste rularea rapida de demo
    print(f"Se vor procesa {len(files_to_process)} imagini")

    areas, color_imgs, hashes = parallel_process_images(files_to_process)
    shape_imgs = load_shape_images(areas)
    sim_matrix = build_similarity_matrix(shape_imgs, areas, color_imgs, hashes)
    groups = group_from_similarity(sim_matrix, shape_imgs.keys())

    if os.path.exists(OUTPUT_FOLDER): #sterge orice folder numit asa, pentru a evita scrierea peste clasele unei rulari anterioare
        shutil.rmtree(OUTPUT_FOLDER)

    save_groups(groups)
    print(f"\nFinalizat! Grupuri salvate în: {OUTPUT_FOLDER}")
