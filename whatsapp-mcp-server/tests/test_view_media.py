"""Tool-level media preview coverage."""

import base64

import main
import media_preview


def test_view_media_returns_mcp_image_content(tmp_path, monkeypatch):
    source = tmp_path / "photo.png"
    source.write_bytes(b"source-bytes")
    monkeypatch.setattr(main, "whatsapp_download_media", lambda *_args: str(source))
    monkeypatch.setattr(media_preview, "render_preview", lambda *_args, **_kwargs: (b"image-bytes", "png"))

    image = main.view_media("message-1", "chat@g.us")

    assert image.to_image_content().type == "image"
    assert image.to_image_content().mimeType == "image/png"
    assert base64.b64decode(image.to_image_content().data) == b"image-bytes"


def test_view_media_rejects_invalid_dimension_before_download(monkeypatch):
    def should_not_download(*_args):
        raise AssertionError("invalid input must not download media")

    monkeypatch.setattr(main, "whatsapp_download_media", should_not_download)

    result = main.view_media("message-1", "chat@g.us", max_dimension=2049)

    assert result == {
        "success": False,
        "message": "max_dimension must be an integer from 1 to 2048 pixels",
    }
