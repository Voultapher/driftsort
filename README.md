# Driftsort

Driftsort is a fast, generic robust stable sort, designed to be the new Rust
standard library `slice::sort`. It is derived from glidesort, meaning that it's
fundamentally a hybrid between quick- and mergesort. Mergesort, or more
specifically, powersort, is used to take advantage of pre-existing ascending or
descending runs. For unordered input (segments) quicksort is used for its fast
average case and optimal performance on inputs with many duplicates, similar to
pdqsort.

This implementation is designed to be robust and highly adaptive to real world
patterns, while putting reasonable limits on the code size.

## Result

A comprehensive analysis of driftsort can be found [here](https://github.com/Voultapher/sort-research-rs/blob/main/writeup/driftsort_introduction/text.md).

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

## Contributing

Please respect the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) when contributing.

## Authors

* **Orson Peters** - [orlp](https://github.com/orlp)
* **Lukas Bergdoll** - [Voultapher](https://github.com/Voultapher)

See also the list of [contributors](https://github.com/Voultapher/driftsort/contributors)
who participated in this project.

## License

This project is dual licensed under the MIT license and the Apache License,
version 2.0 - see the [LICENSE.md](LICENSE.md) file for details.
