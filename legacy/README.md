# Legacy breve (museum / reference)

This directory contains the **original** breve sources from
[jonklein/breve](https://github.com/jonklein/breve) (last upstream activity ~2015):

- C++ kernel + OpenGL simulation (`kernel/`, `simulation/`)
- steve language (`steve/`)
- Python 2 class library and demos (`lib/classes/`, `demos/`)
- Autotools build (`configure.ac`, `Makefile.in`)
- wx / Qt / OS X frontends

**Do not use this as the primary product.** Build requirements (Python 2, ODE,
GLUT, ancient autoconf) are heavily bitrotted on modern systems.

The living revival is the Python 3 package at the repository root
(`python/breve/`, `demos/`). Use this tree as:

1. Behavioral reference when porting demos
2. Historical archive of the classic design
3. Optional archaeology if you want a Dockerized 2015-era build later

See `README_SOURCE.m4` for historical build notes.
