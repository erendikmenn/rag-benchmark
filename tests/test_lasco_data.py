import json
import stat
import zipfile

import pytest

from rag_benchmark import lasco_data as ld
from rag_benchmark.multimodal import load_dataset


def fixture():
    gallery = [{"id": i, "path": f"val2014/COCO_val2014_{i:012d}.jpg"} for i in (1, 2, 3)]
    annotations = [
        {
            "qid": 11,
            "query-image": [1, gallery[0]["path"]],
            "query-text": "make it blue",
            "target-image": [2, gallery[1]["path"]],
        }
    ]
    return annotations, gallery


def rows(a, g):
    return ld.build_rows(a, g, expected_gallery=3, expected_queries=1)


def archive(path, gallery):
    with zipfile.ZipFile(path, "w") as z:
        for g in gallery:
            z.writestr(g["path"], b"fake-jpeg-" + str(g["id"]).encode())
        z.writestr("val2014/unneeded.jpg", b"not selected")


def test_rows_no_gold_and_reference_exclusion():
    a, g = fixture()
    c, q, labels, images = rows(a, g)
    assert len(images) == 3 and "text" not in c[0]
    assert q[0]["text"] == "make it blue"
    assert q[0]["metadata"]["exclude_ids"] == ["1"]
    assert "target" not in json.dumps(c + q)
    assert labels == [{"query_id": "11", "corpus_id": "2", "relevance": 1.0}]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "self", "path", "nonstring", "gold"])
def test_bad_labels(mutation):
    a, g = fixture()
    if mutation == "duplicate":
        g[2] = g[1]
    if mutation == "missing":
        a[0]["target-image"] = [4, "val2014/COCO_val2014_000000000004.jpg"]
    if mutation == "self":
        a[0]["target-image"] = a[0]["query-image"]
    if mutation == "path":
        a[0]["query-image"][1] = "../image.jpg"
    if mutation == "nonstring":
        a[0]["query-text"] = None
    if mutation == "gold":
        a[0]["target-caption"] = "leak"
    with pytest.raises(ValueError):
        rows(a, g)


def test_full_counts_default_reject_subset():
    with pytest.raises(ValueError, match="Incomplete"):
        ld.build_rows(*fixture())


def test_offline_import_roundtrip(tmp_path, monkeypatch):
    a, g = fixture()
    ap, gp, zp, target = [tmp_path / n for n in ["a.json", "g.json", "images.zip", "prepared"]]
    ap.write_text(json.dumps(a))
    gp.write_text(json.dumps(g))
    archive(zp, g)
    monkeypatch.setattr(ld, "ANNOTATION_SHA256", ld.sha256(ap))
    monkeypatch.setattr(ld, "GALLERY_SHA256", ld.sha256(gp))
    monkeypatch.setattr(ld, "GALLERY_COUNT", 3)
    monkeypatch.setattr(ld, "QUERY_COUNT", 1)
    result = ld.import_validation(ap, gp, zp, target, images_sha256=ld.sha256(zp))
    assert result["counts"] == {"corpus": 3, "queries": 1, "qrels": 1, "media_files": 3}
    ds = load_dataset(target)
    assert ds.excluded_ids(ds.queries[0]) == {"1"}
    assert ds.model_item(ds.queries[0])["text"] == "make it blue"
    assert ds.model_item(ds.corpus[0]).get("text") is None
    assert len(list((target / "media").iterdir())) == 3
    assert result["sources"][-1]["sha256_attestation"] == "locally_observed"
    with pytest.raises(ValueError, match="Destination"):
        ld.import_validation(ap, gp, zp, target, images_sha256=ld.sha256(zp))


@pytest.mark.parametrize("name", ["../evil", "/evil", "a/../evil", "a\\evil", "a//evil", "./evil"])
def test_unsafe_zip_paths(tmp_path, name):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(name, b"x")
    with zipfile.ZipFile(z) as f, pytest.raises(ValueError):
        ld.extract_required(f, tmp_path / "out", {})
    assert not (tmp_path / "out").exists()


def test_symlink_zip_rejected(tmp_path):
    z = tmp_path / "bad.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(info, b"target")
    with zipfile.ZipFile(z) as f, pytest.raises(ValueError):
        ld.extract_required(f, tmp_path / "out", {})


def test_duplicate_zip_and_crc(tmp_path):
    _, g = fixture()
    z = tmp_path / "bad.zip"
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(z, "w") as f:
            f.writestr(g[0]["path"], b"aaa")
            f.writestr(g[0]["path"], b"bbb")
    with zipfile.ZipFile(z) as f, pytest.raises(ValueError, match="Duplicate"):
        ld.extract_required(f, tmp_path / "out", {"1": g[0]["path"]})
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_STORED) as f:
        f.writestr(g[0]["path"], b"unique-payload")
    raw = z.read_bytes().replace(b"unique-payload", b"broken-payload", 1)
    z.write_bytes(raw)
    out = tmp_path / "crc"
    out.mkdir()
    with zipfile.ZipFile(z) as f, pytest.raises(zipfile.BadZipFile):
        ld.extract_required(f, out, {"1": g[0]["path"]})


def test_download_requires_explicit_media_pin_before_network(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("network called")

    monkeypatch.setattr(ld.urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError, match="SHA"):
        ld.download_sources(tmp_path, include_images=True)


def test_publication_never_replaces_empty_destination(tmp_path):
    source, target = tmp_path / "stage", tmp_path / "target"
    source.mkdir()
    (source / "dataset.json").write_text("ready")
    target.mkdir()
    with pytest.raises(OSError):
        ld.publish_directory(source, target)
    assert source.is_dir() and not list(target.iterdir())


def test_gallery_official_prefix():
    a, g = fixture()
    for row in g:
        row["path"] = "val/" + row["path"]
    assert len(rows(a, g)[0]) == 3


def test_empty_official_modifier_preserved_as_degraded_query():
    a, g = fixture()
    a[0]["query-text"] = "  "
    _, queries, _, _ = rows(a, g)
    assert queries[0]["text"] == "  "
    assert queries[0]["metadata"]["transition_text_empty"] is True


def test_negative_reverse_query_id_preserved():
    a, g = fixture()
    a[0]["qid"] = -11
    assert rows(a, g)[1][0]["id"] == "-11"


def test_extractor_rejects_unsafe_id_even_with_valid_zip(tmp_path):
    _, gallery = fixture()
    z = tmp_path / "images.zip"
    archive(z, gallery)
    with zipfile.ZipFile(z) as source, pytest.raises(ValueError, match="ID"):
        ld.extract_required(source, tmp_path / "out", {"../evil": gallery[0]["path"]})
    assert not (tmp_path / "out").exists()
