from backend.schemas.documents import PageText
from backend.services.text_chunker import chunk_pages, clean_text


def pages_of(*counts):
    return [
        PageText(page_number=i + 1, text=" ".join(f"p{i+1}w{j}" for j in range(n)))
        for i, n in enumerate(counts)
    ]


def run(pages, size=40, overlap=8):
    return chunk_pages(pages, document_id="doc1", filename="f.pdf", chunk_size=size,
                       chunk_overlap=overlap, tokens_per_word=1.0)


def test_clean_text_normalizes_whitespace_and_keeps_paragraphs():
    raw = "Hello   world\u00a0again\r\nsecond line\n\n\n\nNew  para with exam-\nple word"
    out = clean_text(raw)
    assert out == "Hello world again second line\n\nNew para with example word"


def test_metadata_is_preserved_on_every_chunk():
    chunks = run(pages_of(100))
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.document_id == "doc1" and c.filename == "f.pdf"
        assert c.chunk_id == f"doc1:{i:05d}" and c.chunk_index == i
        assert c.page_start == c.page_end == 1 and c.text


def test_chunk_spanning_pages_records_the_page_range():
    chunks = run(pages_of(30, 30), size=40, overlap=5)
    spanning = [c for c in chunks if "p1w29" in c.text and "p2w0" in c.text]
    assert spanning, "a chunk should straddle the page break"
    c = spanning[0]
    assert (c.page_start, c.page_end, c.pages) == (1, 2, [1, 2])
    assert c.page_label == "1-2"


def test_single_page_chunks_are_not_given_a_range():
    chunks = run(pages_of(30, 30), size=40, overlap=5)
    first = chunks[0]
    assert first.page_start == first.page_end == 1 and "p2w0" not in first.text
    assert all(p in (1, 2) for c in chunks for p in c.pages)


def test_every_chunk_page_list_matches_its_words():
    for c in run(pages_of(25, 40, 35), size=30, overlap=6):
        words_pages = {int(w[1]) for w in c.text.split() if w.startswith("p")}
        assert set(c.pages) == words_pages


def test_overlap_and_full_coverage():
    pages = pages_of(120)
    chunks = run(pages, size=40, overlap=10)
    all_words = [f"p1w{j}" for j in range(120)]
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(all_words)
    shared = set(chunks[0].text.split()) & set(chunks[1].text.split())
    assert len(shared) >= 5


def test_chunk_size_is_respected_and_terminates_with_large_overlap():
    chunks = run(pages_of(200), size=40, overlap=39)
    assert all(len(c.text.split()) <= 40 for c in chunks)
    assert len(chunks) < 400


def test_paragraph_boundary_preferred():
    text = " ".join(f"a{i}" for i in range(35)) + "\n\n" + " ".join(f"b{i}" for i in range(35))
    chunks = chunk_pages([PageText(page_number=1, text=text)], document_id="d", filename="f",
                         chunk_size=40, chunk_overlap=5, tokens_per_word=1.0)
    assert chunks[0].text.endswith("a34")


def test_empty_input_gives_no_chunks():
    assert run([]) == []
    assert run([PageText(page_number=1, text="   \n\n  ")]) == []
