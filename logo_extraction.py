import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from wand.image import Image as WandImage
import urllib3

SAVE_FOLDER = "logos"
os.makedirs(SAVE_FOLDER, exist_ok=True)
MAX_THREADS = 20
TIMEOUT = 20
RETRIES = 2

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'
}

def convert_svg_to_png(svg_path, png_path):
    try:
        with WandImage(filename=svg_path) as img:
            img.format = 'png'
            img.save(filename=png_path)
        os.remove(svg_path)
        print(f"[SVG → PNG] {os.path.basename(png_path)}")
        return True
    except Exception as e:
        print(f"[!] Failed SVG → PNG: {svg_path} — {e}")
        return False

def try_request(url):
    for attempt in range(RETRIES):
        try:
            return requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False, allow_redirects=True)
        except requests.exceptions.Timeout:
            if attempt == RETRIES - 1:
                raise
        except requests.exceptions.RequestException:
            return None
    return None

def fetch_logo_from_html(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    candidates = []

    candidates += [link.get('href') for link in soup.find_all('link', rel=lambda x: x and 'icon' in x)]
    candidates += [meta.get('content') for meta in soup.find_all('meta', property='og:image')]

    for img in soup.find_all('img'):
        attributes = (
            str(img.get('id', '')) +
            ' '.join(map(str, img.get('class', []))) +
            str(img.get('alt', ''))
        ).lower()
        if 'logo' in attributes and img.get('src'):
            candidates.append(img.get('src'))

    for url in candidates:
        if url:
            full_url = urljoin(base_url, url)
            if full_url.startswith(('http://', 'https://')) and 'etc' not in full_url:
                return full_url
    return None

def try_clearbit(domain):
    logo_url = f"https://logo.clearbit.com/{domain}"
    try:
        response = requests.get(logo_url, headers=HEADERS, timeout=8)
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            return logo_url
    except:
        return None
    return None

def try_favicon(domain):
    for protocol in ["https://", "http://"]:
        favicon_url = f"{protocol}{domain}/favicon.ico"
        try:
            response = requests.get(favicon_url, headers=HEADERS, timeout=8, verify=False)
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                return favicon_url
        except:
            continue
    return None

def save_logo(logo_url, index, domain):
    logo_response = try_request(logo_url)
    if not logo_response or not logo_response.ok:
        return False

    ext = os.path.splitext(logo_url)[1].split('?')[0].split('&')[0].lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.ico', '.svg', '.webp']:
        ext = '.png'

    filename = f"{index}_{domain.replace('.', '_')}{ext}"
    filepath = os.path.join(SAVE_FOLDER, filename)

    try:
        with open(filepath, 'wb') as f:
            f.write(logo_response.content)
    except Exception as e:
        print(f"[x] Failed to save logo: {domain} — {e}")
        return False

    # Convert SVG → PNG
    if ext == '.svg':
        png_path = filepath.replace('.svg', '.png')
        return convert_svg_to_png(filepath, png_path)

    # Convert ICO → PNG, only if it's valid
    if ext == '.ico':
        try:
            with WandImage(filename=filepath) as img:
                img.format = 'png'
                png_path = filepath.replace('.ico', '.png')
                img.save(filename=png_path)
            os.remove(filepath)  # Delete .ico after converting
            print(f"[ICO → PNG] {os.path.basename(png_path)}")
        except Exception as e:
            print(f"[x] Failed ICO → PNG: {filepath} — {e}")
            os.remove(filepath)  # Remove corrupted ICO
            return False

    return True


def download_logo(index, domain):
    domain = domain.strip().lower()
    html_response = None

    for protocol in ["https://", "http://"]:
        try:
            website_url = f"{protocol}{domain}"
            html_response = try_request(website_url)
            if html_response and html_response.ok:
                break
        except:
            continue

    if html_response and html_response.ok:
        logo_url = fetch_logo_from_html(html_response.text, website_url)
        if logo_url and save_logo(logo_url, index, domain):
            return f"[HTML] {domain}", True

    logo_url = try_clearbit(domain)
    if logo_url and save_logo(logo_url, index, domain):
        return f"[Clearbit] {domain}", True

    logo_url = try_favicon(domain)
    if logo_url and save_logo(logo_url, index, domain):
        return f"[Favicon] {domain}", True

    return f"[x] Failed for {domain}", False

def run_parallel(domains, max_threads=MAX_THREADS):
    results = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_domain = {
            executor.submit(download_logo, idx, domain): domain
            for idx, domain in enumerate(domains)
        }
        for i, future in enumerate(as_completed(future_to_domain), 1):
            domain = future_to_domain[future]
            try:
                message, success = future.result()
                print(f"{i}/{len(domains)} {message}")
                results.append((domain, success))
            except Exception as e:
                print(f"{i}/{len(domains)} [x] Crash on {domain}: {e}")
                results.append((domain, False))
    return results

def main():
    df = pd.read_parquet("logos.snappy.parquet")
    domains = df['domain'].dropna().unique()
    total = len(domains)

    print(f"\nTotal domains to process: {total}\n")

    results = run_parallel(domains)

    success_count = sum(1 for _, ok in results if ok)
    fail_count = total - success_count
    percent = round(success_count / total * 100, 2)

    # Save failures
    failed_domains = [d for d, ok in results if not ok]
    pd.DataFrame(failed_domains, columns=["domain"]).to_csv("failed_logos.csv", index=False)

    print("\n=== DONE ===")
    print(f"Logos downloaded: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success rate: {percent}%")
    print(f"Failed domains saved to failed_logos.csv\n")

if __name__ == '__main__':
    main()