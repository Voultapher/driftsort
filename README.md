# Driftsort

Driftsort is a fast, generic robust stable sort. Designed to be the new Rust
standard library `slice::sort`. Derived from glidesort. Fundamentally it's a
hybrid quick- merge-sort. Using quicksort to sort parts of the input that are
not ascending or descending. And using a powersort based merge-sort to combine
these parts with parts of the input that is already sorted ascending or
descending. Together with pdqsort style common value filtering. This
implementation is designed to be robust and highly adaptive to real world
patterns.

## Goals

- Stable `slice::sort` replacement
- Good performance for real world types and inputs
- Fast to compile
- Small debug footprint
- Good debug performance

## Non-goals

- The fastest integer sort possible
- Same pattern support as glidesort (?)
- Same performance as glidesort
- N / x auxiliary memory support (? - might get for free)

## How to use

TODO

## Contributing

Please respect the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) when contributing.

## Authors

* **Orson Peters** - [orlp](https://github.com/orlp)
* **Lukas Bergdoll** - [Voultapher](https://github.com/Voultapher)

See also the list of [contributors](https://github.com/Voultapher/driftsort/contributors)
who participated in this project.

## License

This project is licensed under the Apache License, Version 2.0 -
see the [LICENSE.md](LICENSE.md) file for details.


