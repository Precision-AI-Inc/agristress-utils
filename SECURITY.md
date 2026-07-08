# Security Policy

## Supported Versions

Security fixes are provided for the latest released version of `pai-agristress-utils`.

Older versions are not guaranteed to receive patches. Please upgrade to the latest release before reporting a vulnerability where practical.

## Reporting a Vulnerability

Do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

Instead, report suspected vulnerabilities privately by emailing:

**michael@precision.ai**

Please include as much of the following as possible:

- A brief description of the issue
- Steps to reproduce, including any relevant input data
- A minimal proof of concept, if available
- The affected package version (`pip show pai-agristress-utils`)
- Python version and operating system
- Any known impact or realistic attack scenario

We will acknowledge receipt within 7 business days and follow up with any questions needed to investigate.

## Scope

Issues are considered security-relevant when they allow:

- Arbitrary code execution triggered by a crafted dataset file or input
- Supply-chain compromise of the package or its release artifacts
- Unauthorized access to data through the HuggingFace download integration

The following are generally out of scope:

- Issues that require pre-existing arbitrary code execution on the host
- Vulnerabilities only affecting unsupported versions
- Reports without enough information to reproduce or assess the issue

## Coordinated Disclosure

Please allow the maintainers reasonable time to investigate and ship a fix before any public disclosure.

If a vulnerability is confirmed, we will:

1. Assess severity and the range of affected versions
2. Prepare and test a fix
3. Release a patched version through PyPI
4. Publish an advisory or release note as appropriate

## Security Updates

Security fixes are distributed through normal package channels. Upgrade with:

```bash
pip install --upgrade pai-agristress-utils
```
