"""Text processing before indexing: parser cleanup, sentence chunking, BM25 tokens."""

import contextlib
import io
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from eval.corpus import IndexedDoc
from services import embedder, parser
from services.embedder import CHUNK_OVERLAP, CHUNK_SIZE, _tokenize_for_bm25, chunk_text

USER = "user-a"


def squash(text: str) -> str:
    return " ".join(text.split())


# ── BM25 tokenizer ──────────────────────────────────────────────────────────

class Bm25TokenizerTests(unittest.TestCase):
    def test_joined_terms_are_kept_whole_and_split(self) -> None:
        self.assertEqual(_tokenize_for_bm25("IL-6"), ["il-6", "il", "6"])
        self.assertEqual(_tokenize_for_bm25("tools/list"), ["tool/list", "tool", "list"])
        self.assertEqual(_tokenize_for_bm25("COVID-19"), ["covid-19", "covid", "19"])

    def test_alphanumeric_terms_are_not_stemmed(self) -> None:
        self.assertEqual(_tokenize_for_bm25("p53 TP53 2020s h2o"), ["p53", "tp53", "2020s", "h2o"])

    def test_other_words_are_porter_stemmed(self) -> None:
        self.assertEqual(_tokenize_for_bm25("Proteins running regulated"), ["protein", "run", "regul"])

    def test_stop_words_are_dropped(self) -> None:
        self.assertEqual(_tokenize_for_bm25("The levels of the protein"), ["level", "protein"])
        self.assertEqual(_tokenize_for_bm25("and/or"), ["and/or"])  # the whole term still counts
        self.assertEqual(_tokenize_for_bm25("what is the"), [])

    def test_lowercases_and_ignores_punctuation(self) -> None:
        self.assertEqual(_tokenize_for_bm25("Invoice #42, (Paid)."), ["invoic", "42", "paid"])


class Bm25RetrievalTests(unittest.TestCase):
    """The tokenizer as BM25 uses it: queries match documents across surface forms."""

    def setUp(self) -> None:
        texts = {
            "il6": "Serum IL-6 concentrations were elevated in severe cases.",
            "p53": "The p53 protein suppresses tumour growth.",
            "mcp": "Clients call tools/list to discover the server's tools.",
            "other": "Interleukin levels are discussed in chapter four.",
        }
        meta = lambda page: {  # noqa: E731
            "doc_id": "doc-a", "user_id": USER, "doc_name": "a.pdf", "page_number": page, "image_filename": "",
        }
        collection = MagicMock()
        collection.get.return_value = {
            "ids": list(texts),
            "documents": list(texts.values()),
            "metadatas": [meta(i) for i in range(1, len(texts) + 1)],
        }
        self.patch = patch.object(embedder, "collection", collection)
        self.patch.start()
        embedder._invalidate_bm25_index()

    def tearDown(self) -> None:
        self.patch.stop()
        embedder._invalidate_bm25_index()

    def search(self, query: str) -> list[str]:
        with contextlib.redirect_stdout(io.StringIO()):
            return [h["chunk_id"] for h in embedder._search_bm25(query, 5, USER, None)]

    def test_hyphenated_term_matches_with_or_without_the_hyphen(self) -> None:
        for query in ("IL-6 concentration", "il 6 concentrations", "IL6? no: IL-6"):
            with self.subTest(query=query):
                self.assertEqual(self.search(query)[0], "il6")

    def test_stemmed_query_matches_inflected_document(self) -> None:
        self.assertEqual(self.search("proteins suppressing tumours")[0], "p53")
        self.assertEqual(self.search("p53")[0], "p53")

    def test_slash_term(self) -> None:
        self.assertEqual(self.search("tools/list")[0], "mcp")

    def test_stop_word_only_query_finds_nothing(self) -> None:
        self.assertEqual(self.search("what is the"), [])


# ── chunking ────────────────────────────────────────────────────────────────

# ~30-char sentences: each fits within CHUNK_OVERLAP.
PROSE = " ".join(f"Fact {n} is about topic {n % 7}." for n in range(1, 70))


class ChunkTextTests(unittest.TestCase):
    def test_empty_and_short_text(self) -> None:
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("  \n\n  "), [])
        self.assertEqual(chunk_text("  One sentence.  "), ["One sentence."])

    def test_chunks_are_slices_that_end_at_sentence_ends(self) -> None:
        chunks = chunk_text(PROSE)

        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), CHUNK_SIZE)
            self.assertIn(chunk, PROSE)
            self.assertRegex(chunk, r"^Fact \d+ ")
            self.assertRegex(chunk, r"Fact \d+ is about topic \d\.$")

    def test_consecutive_chunks_overlap_by_whole_sentences_up_to_the_limit(self) -> None:
        chunks = chunk_text(PROSE)
        for previous, chunk in zip(chunks, chunks[1:]):
            # Every sentence here is under CHUNK_OVERLAP chars, so the overlap is the last one.
            last_sentence = previous[previous.rindex("Fact"):]
            self.assertLessEqual(len(last_sentence), CHUNK_OVERLAP)
            self.assertTrue(chunk.startswith(last_sentence + " "), (last_sentence, chunk[:60]))

    def test_long_sentences_overlap_at_a_word_boundary(self) -> None:
        sentence = "This sentence keeps going with many words so that it is much longer than the overlap. "
        chunks = chunk_text(sentence * 12)
        for previous, chunk in zip(chunks, chunks[1:]):
            overlap = next(n for n in range(CHUNK_OVERLAP, 0, -1) if previous.endswith(chunk[:n]))
            self.assertLessEqual(overlap, CHUNK_OVERLAP)
            self.assertGreater(overlap, 0)
            self.assertTrue(previous[-overlap - 1].isspace())  # starts at a word
            self.assertTrue(chunk[overlap].isspace())

    def test_paragraph_end_is_preferred_once_the_chunk_is_half_full(self) -> None:
        first = " ".join(["Alpha paragraph sentence about cells."] * 8)  # ~300 chars
        second = " ".join(["Beta paragraph sentence about genes."] * 8)
        chunks = chunk_text(first + "\n\n" + second)

        self.assertEqual(chunks[0], first)
        self.assertTrue(chunks[1].endswith(second))

    def test_short_paragraphs_are_packed_together(self) -> None:
        text = "Title line\n\nA short intro.\n\nAnother short paragraph."
        self.assertEqual(chunk_text(text), [text])

    def test_unpunctuated_text_is_cut_between_words(self) -> None:
        words = " ".join(f"word{n}" for n in range(400))
        chunks = chunk_text(words)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), CHUNK_SIZE)
            self.assertRegex(chunk, r"^word\d+")
            self.assertRegex(chunk, r"word\d+$")
            self.assertIn(f" {chunk} ", f" {words} ")  # whole words only

    def test_table_rows_are_cut_at_line_breaks(self) -> None:
        table = "\n".join(f"row {n} | {n * 3} | ok" for n in range(80))
        for chunk in chunk_text(table):
            self.assertLessEqual(len(chunk), CHUNK_SIZE)
            self.assertRegex(chunk.split("\n")[-1], r"^row \d+ \| \d+ \| ok$")

    def test_a_single_huge_token_is_hard_cut(self) -> None:
        chunks = chunk_text("x" * 1200)
        # No word boundary to start an overlap at, so the pieces just tile the text.
        self.assertEqual([len(c) for c in chunks], [CHUNK_SIZE, CHUNK_SIZE, 200])
        self.assertEqual("".join(chunks), "x" * 1200)

    def test_page_can_be_rebuilt_from_its_chunks(self) -> None:
        text = PROSE + "\n\n" + "\n".join(f"row {n} | ok" for n in range(60)) + "\n\nThe end."
        doc = IndexedDoc("a.pdf", "id")
        doc.chunks[1] = list(enumerate(chunk_text(text)))

        self.assertEqual(squash(doc.page_text(1)), squash(text))

    def test_legacy_fixed_size_chunks_still_rebuild(self) -> None:
        legacy = []
        start = 0
        while start < len(PROSE):
            legacy.append(PROSE[start:start + CHUNK_SIZE])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        doc = IndexedDoc("a.pdf", "id")
        doc.chunks[1] = list(enumerate(legacy))

        self.assertEqual(doc.page_text(1), PROSE)


# ── parser cleanup ──────────────────────────────────────────────────────────

class CleanPageTextTests(unittest.TestCase):
    def test_words_hyphenated_across_lines_are_rejoined(self) -> None:
        self.assertEqual(parser.clean_page_text("an inter-\nnational study"), "an international study")
        self.assertEqual(parser.clean_page_text("con-  \n  tinued"), "continued")

    def test_real_hyphens_split_across_lines_are_kept(self) -> None:
        self.assertEqual(parser.clean_page_text("serum IL-\n6 levels"), "serum IL-6 levels")
        self.assertEqual(parser.clean_page_text("COVID-\n19"), "COVID-19")

    def test_ordinary_line_ends_and_dashes_are_untouched(self) -> None:
        self.assertEqual(parser.clean_page_text("first line\nsecond line"), "first line\nsecond line")
        self.assertEqual(parser.clean_page_text("a well-known fact"), "a well-known fact")
        self.assertEqual(parser.clean_page_text("dash -\nthen"), "dash -\nthen")

    def test_whitespace_is_normalised(self) -> None:
        raw = "  Total:\t\t 42   units \r\n\r\n\r\n\r\nNext para­ graph​  "
        self.assertEqual(parser.clean_page_text(raw), "Total: 42 units\n\nNext para graph")


def page(header: str, body: str, footer: str) -> str:
    return "\n".join(part for part in (header, body, footer) if part)


class StripRepeatedLinesTests(unittest.TestCase):
    def test_running_header_and_page_number_footer_are_removed(self) -> None:
        pages = [
            page("ACME Corp - Confidential", f"Body text of page {n}.\nMore content here.", f"Page {n} of 4")
            for n in range(1, 5)
        ]
        cleaned = parser.strip_repeated_lines(pages)

        for n, text in enumerate(cleaned, start=1):
            self.assertEqual(text, f"Body text of page {n}.\nMore content here.")

    def test_lines_on_half_the_pages_or_fewer_stay(self) -> None:
        pages = [page("Chapter 1", "Alpha.", ""), page("Chapter 1", "Beta.", ""),
                 page("", "Gamma.", ""), page("", "Delta.", "")]
        self.assertEqual(parser.strip_repeated_lines(pages), pages)

    def test_repeated_lines_in_the_middle_of_a_page_stay(self) -> None:
        body = "\n".join(["Intro."] + [f"Detail {n}." for n in range(4)] + ["Repeated middle line"]
                         + [f"Detail {n}." for n in range(4, 8)] + ["Outro."])
        pages = [f"Header {n}\n{body}" for n in range(1, 5)]

        cleaned = parser.strip_repeated_lines(pages)

        for text in cleaned:
            self.assertIn("Repeated middle line", text)
            self.assertNotIn("Header", text)

    def test_needs_three_pages_with_text(self) -> None:
        pages = [page("Letterhead", "One.", ""), page("Letterhead", "Two.", ""), "", ""]
        self.assertEqual(parser.strip_repeated_lines(pages), pages)


class FakePage:
    def __init__(self, text: str, tables: list | None = None) -> None:
        self._text, self._tables = text, tables or []

    def extract_text(self) -> str:
        return self._text

    def extract_tables(self) -> list:
        return self._tables

    def to_image(self, resolution: int):
        return SimpleNamespace(original=Image.new("RGB", (4, 4)))


class ParsePdfCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_pages_are_cleaned_and_tables_survive_header_removal(self) -> None:
        table = [["Item", "Qty"], ["Widget", "3"]]
        pages = [
            FakePage(f"Quarterly   Report\nThe inter-\nnational results for part {n}.\n- {n} -", tables=[table])
            for n in range(1, 4)
        ]
        pdf = MagicMock(pages=pages)
        pdf.__enter__.return_value = pdf

        with (
            patch.object(parser.pdfplumber, "open", return_value=pdf),
            patch.object(parser, "_strip_hidden_text", return_value=0),
            contextlib.redirect_stdout(io.StringIO()),
            tempfile.TemporaryDirectory() as image_dir,
        ):
            result = await parser._parse_pdf("x.pdf", "doc", image_dir)

        for n, parsed in enumerate(result, start=1):
            # Header and page-number footer gone; the table (the same on every
            # page) is kept, after the body, as its own paragraph.
            self.assertEqual(parsed["text"], f"The international results for part {n}.\n\nItem | Qty\nWidget | 3")
            self.assertNotIn("_table_texts", parsed)


if __name__ == "__main__":
    unittest.main()
