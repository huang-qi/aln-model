# Security Policy

## Model artifact safety

Never load untrusted joblib files. `joblib` deserialization can execute
arbitrary code, so treat every model artifact as executable content. Load an
artifact only when you trust its source and have verified its integrity.

## Reporting a vulnerability

Repository owners or administrators who can access GitHub Security Advisories
may report a vulnerability through a
[private advisory](https://github.com/huang-qi/aln-model/security/advisories/new).

Other collaborators and reporters should contact the repository owner or
administrator through the private channel through which repository access was
granted. Do not open a normal GitHub issue or otherwise disclose the
vulnerability publicly. Include reproduction steps, affected versions, impact,
and any suggested mitigation when possible.
