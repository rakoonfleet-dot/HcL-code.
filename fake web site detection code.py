import re
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier


class FakeWebsiteDetector:
    """Heuristic + Machine Learning classifier for detecting suspicious/fake URLs."""

    SUSPICIOUS_KEYWORDS = {
        "login", "verify", "update", "secure", "account",
        "banking", "free", "bonus", "confirm", "suspend",
        "signin", "wallet", "gift", "urgent"
    }

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._train_sample_model()

    @staticmethod
    def _has_ip_address(hostname: str) -> int:
        if not hostname:
            return 0
        # Matches IPv4 addresses used directly as a host
        return 1 if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", hostname) else 0

    def extract_features(self, url: str) -> list[float]:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        hostname = (parsed.hostname or "").lower()
        path_and_query = f"{parsed.path}?{parsed.query}".lower()
        full_url = url.lower()

        url_length = len(url)
        hostname_length = len(hostname)
        has_ip = self._has_ip_address(hostname)
        dot_count = hostname.count(".")
        hyphen_count = hostname.count("-")
        has_at_symbol = 1 if "@" in full_url else 0
        uses_https = 1 if parsed.scheme == "https" else 0
        keyword_hits = sum(
            1 for kw in self.SUSPICIOUS_KEYWORDS if kw in path_and_query or kw in hostname
        )

        return [
            url_length,
            hostname_length,
            has_ip,
            dot_count,
            hyphen_count,
            has_at_symbol,
            uses_https,
            keyword_hits,
        ]

    def _train_sample_model(self):
        """
        Trains on a small starter dataset for demonstration.
        Label: 1 = Fake/Phishing, 0 = Legitimate
        Replace or expand this dataset with real labeled data (e.g., PhishTank + Tranco)
        for production use.
        """
        sample_urls = [
            # Legitimate examples (0)
            ("https://www.google.com", 0),
            ("https://en.wikipedia.org/wiki/Machine_learning", 0),
            ("https://github.com/scikit-learn/scikit-learn", 0),
            ("https://www.python.org/downloads/", 0),
            ("https://docs.python.org/3/library/urllib.parse.html", 0),
            ("https://www.amazon.in/", 0),
            ("https://developer.mozilla.org/en-US/docs/Web/HTTP", 0),
            ("https://stackoverflow.com/questions", 0),

            # Fake / Phishing-style examples (1)
            ("http://192.168.1.45/login/verify-account", 1),
            ("http://secure-login-update-bank-account.xyz/confirm", 1),
            ("http://paypa1-signin.verify-user-security.com/account", 1),
            ("http://free-gift-bonus-wallet-claim.tk/login", 1),
            ("http://accounts.google.com.security-check-urgent.cn/signin", 1),
            ("http://www.bank-suspend-verify@malicious-host.info/update", 1),
            ("http://my-secure-banking-login-alert-2026.pw/verify", 1),
            ("http://update-your-wallet-and-confirm-identity.top/bonus", 1),
        ]

        X = [self.extract_features(u) for u, _ in sample_urls]
        y = [label for _, label in sample_urls]
        self.model.fit(X, y)

    def predict(self, url: str) -> dict:
        features = [self.extract_features(url)]
        prediction = int(self.model.predict(features)[0])
        fake_probability = float(self.model.predict_proba(features)[0][1])

        return {
            "url": url,
            "status": "FAKE / SUSPICIOUS" if prediction == 1 else "LIKELY LEGITIMATE",
            "risk_score": round(fake_probability * 100, 2),
        }


if __name__ == "__main__":
    detector = FakeWebsiteDetector()

    test_urls = [
        "https://www.python.org/doc/",
        "http://secure-login-verify-bank-account.xyz/update",
        "http://203.0.113.55/signin/confirm",
        "https://www.github.com/trending",
    ]

    for target in test_urls:
        result = detector.predict(target)
        print(
            f"{result['url']} -> {result['status']} (Risk Score: {result['risk_score']}%)"
        )