import os
import re
from pathlib import Path
from typing import Generator

from detect_secrets import SecretsCollection
from detect_secrets.plugins.base import RegexBasedDetector
from detect_secrets.settings import transient_settings

path = os.path.realpath(__file__)


class DAPIDetector(RegexBasedDetector):
    """Scans for databricks api tokens."""
    secret_type = 'Databricks API Token'

    denylist = (
        re.compile(
            r'dapi[\w]*',
            re.IGNORECASE,
        ),
        re.compile(
            r'dkea[\w]*',
            re.IGNORECASE,
        ),
    )

    def analyze_string(self, string: str) -> Generator[str, None, None]:
        for regex in self.denylist:
            for match in regex.findall(string):
                if len(match) < 10:  # ignore short matches
                    continue
                if isinstance(match, tuple):
                    for submatch in filter(bool, match):
                        yield submatch
                else:
                    yield match


all_plugins = """ArtifactoryDetector
AWSKeyDetector
AzureStorageKeyDetector
BasicAuthDetector
CloudantDetector
DiscordBotTokenDetector
GitHubTokenDetector
Base64HighEntropyString
HexHighEntropyString
IbmCloudIamDetector
IbmCosHmacDetector
JwtTokenDetector
MailchimpDetector
NpmDetector
PrivateKeyDetector
SendGridDetector
SlackDetector
SoftlayerDetector
SquareOAuthDetector
StripeDetector
TwilioKeyDetector""".split("\n")


def get_all_files(directory):
    return [str(file) for file in Path(directory).rglob('*') if file.is_file()]


def scan_in_directory(directory: str) -> Generator[str, None, None]:
    files = get_all_files(directory)
    secrets = SecretsCollection()
    with transient_settings({
        # Only run scans with only these plugins.
        # This format is the same as the one that is saved in the generated baseline.
        'plugins_used': [
            {
                'name': 'DAPIDetector',
                'path': f"file://{path}"
            },
            *[{"name": p_name} for p_name in all_plugins]
        ],

    }):
        secrets.scan_files(
            *files, num_processors=None
        )
        for k, l in secrets.json().items():
            for s in l:
                relative_file = Path(s['filename']).relative_to(directory)
                yield f"POTENTIAL SECRET FOUND - [{s['type']}][LINENO:{s['line_number']}]: {relative_file}"
