import cv2
import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def png(dark_image):
    ok, buf = cv2.imencode(".png", dark_image)
    return buf.tobytes()


def post(client, png, **form):
    from io import BytesIO

    return client.post("/api/enhance", data={"image": (BytesIO(png), "dark.png"), **form},
                       content_type="multipart/form-data")


def test_status(client, demo_mode):
    assert client.get("/api/status").get_json() == {"mode": "demo"}


def test_index(client):
    assert client.get("/").status_code == 200


def test_enhance_ok(client, demo_mode, png):
    r = post(client, png, strength="0.5", brightness="0.6")
    assert r.status_code == 200
    body = r.get_json()
    assert all(v.startswith("data:image/png;base64,") for v in body["images"].values())


def test_missing_image(client):
    assert client.post("/api/enhance").status_code == 400


@pytest.mark.parametrize("form", [{"strength": "abc"}, {"strength": "nan"}, {"brightness": "inf"}])
def test_bad_controls(client, demo_mode, png, form):
    assert post(client, png, **form).status_code == 400


def test_bad_image(client):
    from io import BytesIO

    r = client.post("/api/enhance", data={"image": (BytesIO(b"not an image"), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
