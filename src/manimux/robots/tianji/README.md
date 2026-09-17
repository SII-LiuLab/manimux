# Tianji compatibility imports

The implementation now lives in
[`embodiments/robot/tianji_taccap`](../../embodiments/robot/tianji_taccap/README.md).
Official arm kinematics and the unmodified Marvin SDK live in
`embodiments/arm/tianji/`.

Python imports from this directory forward to the same implementation modules.
This preserves class identity and the process-wide SDK locks. Vendor files and
libraries are installed only at the new location; direct filesystem references
must use that location.

This directory does not restore the legacy `tianji_dual` runtime entry.
