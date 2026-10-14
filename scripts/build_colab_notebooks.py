"""Generate three output-free Colab notebooks pinned to a reviewed source commit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(True)}


def code(text):
    source = textwrap.dedent(text).strip() + "\n"
    compile(source, "<colab-cell>", "exec")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


def notebook(job, source_ref):
    job_id = job['id']
    dataset = job['dataset']
    cells = [markdown(f"""
        # RAG benchmark — {job_id}

        **{dataset['query_count']} sorgu / {dataset['corpus_count']} sayfa.**
        Bu notebook B/G/E/N/S/J için iki aday bütçesinde 126 tam arama koşulu hedefler.
        CUDA BF16 ayrı bir koşuldur; eski MPS sonuçlarıyla otomatik birleştirilmez.
        Model/GPU başarısı henüz ölçülmedi. Bu ilk dalga Gemma, QA veya reranker içermez.

        1. GPU çalışma zamanı seçin: doğal BF16 desteği olan GPU gerekir (ör. L4/A100).
        2. Size verilen özel `{job_id}.tar` paketini Drive → `MyDrive/rag-benchmark-colab/input/` içine koyun.
        3. Kurulum, Drive, veri doğrulama ve plan hücrelerini çalıştırın.
        4. Tam koşuyu başlatmak için son hücrede `RUN_FULL = True` yapın.

        Her notebook ayrı Colab çalışma zamanında çalışır. Aynı notebooku iki oturumda başlatmayın.
        Ham veri, sorgu, model çıktısı veya hata günlüklerini notebook çıktısına yapıştırmayın.
        [Çalıştırma ve SSH yönetim rehberi](https://github.com/erendikmenn/rag-benchmark/blob/main/docs/colab-running.tr.md)
    """), code(f'''
        from pathlib import Path
        import json
        import os
        import subprocess
        import sys

        JOB = {job_id!r}
        SOURCE_REF = {source_ref!r}
        REPO = Path('/content/rag-benchmark')
        WORKSPACE = Path('/content/rag-work') / JOB
        DATASET = Path('/content/rag-data') / JOB
        DRIVE_ROOT = Path('/content/drive/MyDrive/rag-benchmark-colab')
        PRIVATE = Path('/content/rag-private')
        PRIVATE.mkdir(mode=0o700, exist_ok=True)
        os.umask(0o077)

        def checked(args, label, cwd=None):
            with (PRIVATE / 'setup.log').open('ab') as log:
                result = subprocess.run(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(label + ' başarısız; özel setup.log incelenmeli.')
            print(label + ': tamam')

        checked([sys.executable, '-m', 'pip', 'install', 'uv==0.10.9'], 'Kurulum aracı')
        if not REPO.exists():
            checked(['git', 'clone', 'https://github.com/erendikmenn/rag-benchmark.git', str(REPO)], 'Kaynak kod')
        current = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain'], capture_output=True, text=True, check=True)
        if current.stdout.strip():
            raise RuntimeError('Kaynak ağacında değişiklik var; otomatik üzerine yazılmadı.')
        checked(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_REF], 'Sabit kaynak sürümü')
        assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE_REF
        checked([sys.executable, '-m', 'uv', 'sync', '--frozen', '--all-extras', '--python', '3.12'], 'Kilitli bağımlılıklar', REPO)
        PYTHON = str(REPO / '.venv/bin/python')
        os.environ['HF_HOME'] = '/content/hf-cache'
        print('Kaynak sürümü:', SOURCE_REF)
    '''), markdown("""
        ## GPU kontrolü ve isteğe bağlı model erişimi

        Bu hücre model yüklemez. Doğal BF16 yoksa koşu durur; sessizce FP16 veya CPU kullanılmaz.
        Erişim onayı isteyen Hugging Face modelleri için hesabınızda ilgili koşulları kabul etmiş olmalısınız.
        Gerekirse Colab'ın anahtar simgesinden `HF_TOKEN` adlı secret ekleyin; tokenı hücreye yazmayın.
    """), code(r'''
        from google.colab import userdata

        probe = """import json, torch
        if not torch.cuda.is_available():
            raise SystemExit('GPU çalışma zamanı gerekli.')
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        native = p.major >= 8 and torch.cuda.is_bf16_supported(including_emulation=False)
        print(json.dumps({'gpu': p.name, 'VRAM_GiB': round(p.total_memory / 2**30, 1), 'native_bf16': native}))
        if not native:
            raise SystemExit('Bu plan doğal BF16 gerektiriyor; farklı sayısal koşul otomatik seçilmedi.')
        """
        subprocess.run([PYTHON, '-c', '\n'.join(line[8:] if line.startswith('        ') else line for line in probe.splitlines())], check=True)
        try:
            os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
            print('HF_TOKEN secret ortam değişkenine aktarıldı; değeri gösterilmedi.')
        except userdata.SecretNotFoundError:
            print('HF_TOKEN tanımlı değil; erişim isteyen model indirmeleri başarısız olabilir.')
        except userdata.NotebookAccessError:
            raise RuntimeError('HF_TOKEN secret için notebook erişimini açın.') from None
    '''), code(r'''
        from google.colab import drive
        drive.mount('/content/drive')
        bundles = json.loads((REPO / 'configs/colab-bundle-checksums.json').read_text())['bundles']
        bundle = bundles[JOB]
        archive = DRIVE_ROOT / 'input' / bundle['filename']
        if not archive.is_file():
            raise FileNotFoundError('Özel veri paketini Drive/rag-benchmark-colab/input içine koyun.')
        if not DATASET.exists():
            checked([PYTHON, '-m', 'rag_benchmark.colab_bundle', 'unpack',
                     '--archive', str(archive), '--sha256', bundle['sha256'], '--output', str(DATASET)],
                    'Veri arşivi ve medya hash doğrulaması', REPO)
        else:
            print('Yerel veri dizini var; aşağıdaki plan doğrulaması tüm kaynak kimliğini kontrol edecek.')
        PLAN = REPO / 'configs/colab-retrieval-jobs.json'
        BASE_COMMAND = [PYTHON, '-m', 'rag_benchmark.colab_worker', '--plan', str(PLAN),
                        '--job', JOB, '--dataset', str(DATASET), '--workspace', str(WORKSPACE)]
        subprocess.run(BASE_COMMAND, cwd=REPO, check=True)
    '''), markdown("""
        ## Sabit model dosyalarını hazırla

        Arama adaptörleri yalnız yerel dosyaları açar. Bu hücre BGE-M3, EmbeddingGemma 2 ve
        ColQwen adaptörü + taban modelini kodda sabitlenmiş revision'lardan indirir; model forward yapmaz.
        Bu dosyalar büyük olabilir ve VM diski kullanır. İndirme/erişim hatasında farklı modele geçilmez.
    """), code(r'''
        model_setup = """from rag_benchmark.models import MODEL_DEFAULTS, fetch_model
        from rag_benchmark.multimodal_colqwen import COLQWEN_DEFAULTS, pinned_snapshots
        for name in ('bge', 'embeddinggemma'):
            fetch_model(MODEL_DEFAULTS[name])
        pinned_snapshots(COLQWEN_DEFAULTS, download=True)
        """
        model_setup = '\n'.join(line[8:] if line.startswith('        ') else line for line in model_setup.splitlines())
        checked([PYTHON, '-c', model_setup], 'Sabit model dosyaları', REPO)
    '''), markdown("""
        ## Önceki oturumdan kurtarma (isteğe bağlı)

        Worker her başarılı kanalın ardından özel bir Drive checkpoint'i kaydeder. Son arşivin SHA-256'sı
        ve yolları doğrulanmadan geri yükleme yapılmaz. Python, paketler, CUDA ve GPU modeli değişmişse
        worker devamı reddeder; otomatik kimlik değiştirilmez. Ani VM kaybında son checkpoint'ten sonraki
        çalışma kaybolabilir. Drive dosyalarını başka kişilerle paylaşmayın.
    """), code(r'''
        RESTORE_CHECKPOINT = False  # @param {type:"boolean"}
        CHECKPOINTS = DRIVE_ROOT / 'checkpoints' / JOB
        if RESTORE_CHECKPOINT:
            import hashlib
            import shutil
            import tarfile
            import tempfile
            from pathlib import PurePosixPath
            pointer = json.loads((CHECKPOINTS / (JOB + '-latest.json')).read_text())
            if pointer['job'] != JOB or pointer['protocol'] != 'colab-cuda-bf16-retrieval-v1':
                raise RuntimeError('Checkpoint işi/protokolü uyuşmuyor.')
            name = pointer['archive']
            if Path(name).name != name or not name.startswith(JOB + '-') or not name.endswith('.tar'):
                raise RuntimeError('Checkpoint yolu geçersiz.')
            snapshot = CHECKPOINTS / name
            with snapshot.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual != pointer['sha256']:
                raise RuntimeError('Checkpoint SHA-256 uyuşmuyor.')
            if WORKSPACE.exists():
                raise RuntimeError('Yerel çalışma alanı zaten var; üzerine yazılmadı.')
            WORKSPACE.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=WORKSPACE.parent) as tmp:
                restored = Path(tmp) / 'workspace'
                restored.mkdir(mode=0o700)
                with tarfile.open(snapshot) as tar:
                    seen = set()
                    for entry in tar.getmembers():
                        path = PurePosixPath(entry.name)
                        if (not entry.isfile() or path.is_absolute() or '\\' in entry.name
                                or any(part in ('', '.', '..') for part in entry.name.split('/'))
                                or entry.name in seen):
                            raise RuntimeError('Checkpoint arşiv girdisi geçersiz.')
                        seen.add(entry.name)
                        destination = restored / entry.name
                        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        with tar.extractfile(entry) as source, destination.open('wb') as target:
                            shutil.copyfileobj(source, target)
                        destination.chmod(0o600)
                restored.rename(WORKSPACE)
            print('Checkpoint geri yüklendi; worker devam kimliğini ayrıca doğrulayacak.')
    '''), markdown("""
        ## Tam koşuyu çalıştır

        `RUN_FULL=True` gerçek GPU kullanımını başlatır. Önceki aynı koşula devam ediyorsanız
        `RESUME=True` seçin. Bir hata olduğunda sonraki kanal başlatılmaz; otomatik tekrar döngüsü yoktur.
        Durdur düğmesi mevcut alt süreci kapatır. Yeni deneme öncesinde özel durum kaydı incelenmelidir.

        Sonuçlar `review/` altında denetim adayıdır. Tam sorgu/126 hücre kontrolü bağımsız metrik
        ve kaynak denetiminin yerine geçmez; ana tamamlanma oranı otomatik artmaz.
    """), code(r'''
        RUN_FULL = False  # @param {type:"boolean"}
        RESUME = False  # @param {type:"boolean"}
        if RUN_FULL:
            import signal
            command = BASE_COMMAND + ['--execute', '--checkpoint-dir', str(CHECKPOINTS)]
            if RESUME:
                command.append('--resume')
            child = subprocess.Popen(command, cwd=REPO, start_new_session=True)
            try:
                exit_code = child.wait()
            except KeyboardInterrupt:
                child.send_signal(signal.SIGTERM)
                print('Durdurma istendi; alt süreç kapanışı ve checkpoint bekleniyor.')
                child.wait()
                raise
            if exit_code:
                raise RuntimeError('Worker durdu. Özel state.json ve günlükler denetlenmeden yeniden başlatmayın.')
        else:
            print('Plan hazır; GPU deneyi başlatılmadı. RUN_FULL=True ile açıkça başlatılır.')
    '''), code(r'''
        state_path = WORKSPACE / 'state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            print(json.dumps({'job': JOB, 'status': state['status'], 'completed_stages': state['completed']}, indent=2))
            print('Özel sonuçlar:', str(WORKSPACE / 'review'))
            print('Drive checkpoint dizini:', str(CHECKPOINTS))
        else:
            print('Bu notebook henüz gerçek deney üretmedi.')
    ''')]
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'colab': {'name': job_id + '.ipynb', 'provenance': [], 'private_outputs': True},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'rag_benchmark': {'source_ref': source_ref, 'job': job_id, 'execution_status': 'not_executed'}},
        'cells': [dict(cell, id=f'cell-{i:02d}') for i, cell in enumerate(cells)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ref', required=True)
    parser.add_argument('--output', type=Path, default=Path('notebooks/colab'))
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{40}', args.source_ref):
        parser.error('source-ref must be a full Git commit SHA')
    plan = json.loads(Path('configs/colab-retrieval-jobs.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for job in plan['jobs']:
        (args.output / (job['id'] + '.ipynb')).write_text(json.dumps(notebook(job, args.source_ref), ensure_ascii=False, indent=2) + '\n')
    print(f"Prepared {len(plan['jobs'])} notebooks; no runtime executed.")


if __name__ == '__main__':
    main()
