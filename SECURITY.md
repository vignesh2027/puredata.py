# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✓         |

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security vulnerabilities.

Report vulnerabilities privately by opening a [GitHub Security Advisory](https://github.com/vignesh2027/puredata.py/security/advisories/new).

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 48 hours. We will work with you to understand and resolve the issue before any public disclosure.

## Scope

puredata processes user-provided data files. The primary security considerations are:

- **File path traversal** — puredata only reads files at paths you provide explicitly
- **Pickle deserialisation** — puredata does not use pickle for any user data
- **Arbitrary code execution** — puredata does not evaluate user-provided code except through the custom rules API, which is opt-in and runs in your own process

## Dependencies

puredata depends on pandas, numpy, scipy, scikit-learn, and other well-maintained libraries. Keep dependencies up to date via `pip install --upgrade puredata`.
