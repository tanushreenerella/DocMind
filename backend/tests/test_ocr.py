"""Gemini OCR: throttling, 429/503 retry with retryDelay, and per-page failure flags."""

import contextlib
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import errors as genai_errors
from PIL import Image

from services import parser


def _api_error(code: int, retry_delay: str | None = None) -> genai_errors.APIError:
    status = {429: "RESOURCE_EXHAUSTED", 503: "UNAVAILABLE", 400: "INVALID_ARGUMENT"}[code]
    details = []
    if retry_delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    body = {"error": {"code": code, "message": "nope", "status": status, "details": details}}
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, body)


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class SlidingWindowLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_once_the_window_is_full(self) -> None:
        clock = FakeClock()
        limiter = parser.SlidingWindowLimiter(2, period=60, clock=clock, sleep=clock.sleep)

        await limiter.acquire()
        clock.now = 10
        await limiter.acquire()
        self.assertEqual(clock.sleeps, [])

        await limiter.acquire()  # third call at t=10: waits until the t=0 call leaves the window
        self.assertEqual(clock.sleeps, [50])
        self.assertEqual(clock.now, 60)

    async def test_spread_out_calls_never_wait(self) -> None:
        clock = FakeClock()
        limiter = parser.SlidingWindowLimiter(5, period=60, clock=clock, sleep=clock.sleep)
        for i in range(20):
            clock.now = i * 12  # exactly 5 per minute
            await limiter.acquire()
        self.assertEqual(clock.sleeps, [])

    def test_default_limit_comes_from_config(self) -> None:
        self.assertEqual(parser._ocr_limiter.max_calls, parser.OCR_MAX_PER_MINUTE)
        self.assertEqual(parser.OCR_MAX_PER_MINUTE, 5)


class OcrRetryTests(unittest.IsolatedAsyncioTestCase):
    async def _ocr(self, side_effect) -> tuple[str | Exception, MagicMock, AsyncMock, AsyncMock]:
        client = MagicMock()
        client.models.generate_content.side_effect = side_effect
        sleep = AsyncMock()
        limiter = MagicMock(acquire=AsyncMock())
        with (
            patch.object(parser, "_get_gemini_client", return_value=client),
            patch.object(parser, "_sleep", sleep),
            patch.object(parser, "_ocr_limiter", limiter),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                result = await parser._ocr_page_with_gemini(Image.new("RGB", (4, 4)))
            except Exception as exc:  # noqa: BLE001 - returned for assertions
                result = exc
        return result, client.models.generate_content, sleep, limiter.acquire

    async def test_429_waits_for_the_servers_retry_delay(self) -> None:
        result, call, sleep, acquire = await self._ocr([_api_error(429, "17s"), _response(" text ")])

        self.assertEqual(result, "text")
        self.assertEqual(call.call_count, 2)
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [17.0])
        # Every attempt, retries included, goes through the throttle.
        self.assertEqual(acquire.await_count, 2)

    async def test_fractional_retry_delay(self) -> None:
        _, _, sleep, _ = await self._ocr([_api_error(429, "1.5s"), _response("ok")])
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [1.5])

    async def test_503_without_retry_info_backs_off_exponentially_then_gives_up(self) -> None:
        result, call, sleep, _ = await self._ocr(_api_error(503))

        self.assertIsInstance(result, genai_errors.ServerError)
        self.assertEqual(call.call_count, parser.OCR_MAX_ATTEMPTS)
        self.assertEqual(parser.OCR_MAX_ATTEMPTS, 4)
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [2.0, 4.0, 8.0])

    async def test_huge_retry_delay_is_capped(self) -> None:
        _, _, sleep, _ = await self._ocr([_api_error(429, "3600s"), _response("ok")])
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [60.0])

    async def test_other_errors_are_not_retried(self) -> None:
        result, call, sleep, _ = await self._ocr(_api_error(400))

        self.assertIsInstance(result, genai_errors.ClientError)
        self.assertEqual(call.call_count, 1)
        sleep.assert_not_awaited()


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text

    def extract_tables(self) -> list:
        return []

    def to_image(self, resolution: int):
        return SimpleNamespace(original=Image.new("RGB", (4, 4)))


class FailedPageFlagTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_pages_whose_ocr_failed_are_flagged(self) -> None:
        pages = [
            FakePage("Plenty of embedded text on this first page."),  # no OCR needed
            FakePage(""),  # OCR succeeds
            FakePage(""),  # OCR fails after retries
            FakePage(""),  # OCR returns nothing: blank page, not a failure
        ]
        pdf = MagicMock(pages=pages)
        pdf.__enter__.return_value = pdf
        ocr = AsyncMock(side_effect=["scanned text here", _api_error(503), ""])

        with (
            patch.object(parser.pdfplumber, "open", return_value=pdf),
            patch.object(parser, "_ocr_page_with_gemini", ocr),
            patch.object(parser, "_strip_hidden_text", return_value=0),
            contextlib.redirect_stdout(io.StringIO()),
            tempfile.TemporaryDirectory() as image_dir,
        ):
            result = await parser._parse_pdf("x.pdf", "doc", image_dir)

        self.assertEqual([p["ocr_failed"] for p in result], [False, False, True, False])
        self.assertEqual(result[1]["text"], "scanned text here")
        self.assertEqual(ocr.await_count, 3)


if __name__ == "__main__":
    unittest.main()
