import json
import math
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:8000"


def post_prediction(payload):
    request = Request(
        f"{BASE_URL}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urlopen(request, timeout=10)


# 1. Check that the API is ready.
with urlopen(f"{BASE_URL}/health", timeout=10) as response:
    assert response.status == 200
    assert json.load(response)["status"] == "ready"

print("PASS: API ready")

# 2. Check a valid prediction.
with post_prediction({"range": 500.0}) as response:
    assert response.status == 200
    result = json.load(response)

assert result["range"] == 500.0
assert math.isfinite(result["predicted_logratio"])
assert math.isclose(
    result["predicted_logratio"],
    -0.0703024695766904,
    rel_tol=1e-6,
    abs_tol=1e-8,
)

print("PASS: expected prediction for the current saved model")

# 3. Check that invalid input is rejected.
try:
    with post_prediction({"range": "invalid"}):
        pass
except HTTPError as error:
    assert error.code == 422, f"Unexpected status: {error.code}"
    error.close()
else:
    raise AssertionError("API accepted invalid input")

print("PASS: invalid input rejected")
print("\nAll API checks passed.")
